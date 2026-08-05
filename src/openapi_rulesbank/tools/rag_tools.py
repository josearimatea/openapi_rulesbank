# src/tools/rag_tools.py

"""
RAG tools for indexing and retrieving OpenAPI reference chunks from Qdrant.

Uses the same embedding model and vector store pattern as openapi_chatbotUI:
    - HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2, dim=384)
    - QdrantVectorStore with dense named vector ("text-dense")

The Qdrant collection is created once (via index_openapi_reference()) and
persists across pipeline runs. Extractor and Reflector nodes call
retrieve_chunks() directly — no re-indexing needed.

Functions:
    chunk_document(text)           — splits text into overlapping chunks
    index_openapi_reference(force) — indexes openapi_reference.md into Qdrant
    search_openapi_reference(query, k) — semantic search, returns joined chunk texts
"""

import os
import uuid
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from qdrant_3gpp_rag import (
    QdrantFactory,
    chunk_markdown_by_headers,
    index_segments,
    drop_collection,
    retrieve,
)

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.hardware import device
from openapi_rulesbank.config.settings import (
    QDRANT_HOST, QDRANT_PORT,
    OPENAPI_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    OPENAPI_REFERENCE_RETRIEVE_CHUNKS,
    SPEC_COLLECTION_PREFIX, SPEC_RETRIEVE_CHUNKS,
    SPEC_CHUNK_SIZE, SPEC_CHUNK_OVERLAP,
    GPP_CORPUS_COLLECTIONS, GPP_CORPUS_RETRIEVE_CHUNKS,
    RULES_COLLECTION_PREFIX, RULES_RETRIEVE_CHUNKS,
)
from openapi_rulesbank.tools.document_tools import load_openapi_reference

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Spec RAG — index the main 3GPP document into a temporary Qdrant collection
# and retrieve related sections during extraction. Backed by qdrant_3gpp_rag.
# ---------------------------------------------------------------------------

def _spec_factory() -> QdrantFactory:
    """Shared QdrantFactory for the spec RAG (same host/port/model as this project)."""
    return QdrantFactory.get_instance(
        host=QDRANT_HOST, port=QDRANT_PORT,
        model_name=EMBEDDING_MODEL, device=device,
    )


def _clause_prefix(title: str, levels: int) -> str:
    """Return the leading clause number of a title, to `levels` parts.

    "12.1.1.3 createMOI" with levels=2 -> "12.1". "" if the title has no clause
    number. Generic — no fixed numbering assumed.
    """
    head = title.strip().split(" ", 1)[0]
    parts = head.split(".")
    if not parts or not parts[0].isdigit():
        return ""
    return ".".join(parts[:levels])


def _service_key_from_clause(clause: str) -> str:
    """Return the service key of a level-2 clause: its second number.

    The document lists the same services twice — once per stage — under matching
    clause numbers ("11.3" is the stage 2 of the service whose stage 3 is "12.3"),
    so the second number identifies the service across both blocks.

    Titles cannot play that role: they agree for most services but not all (one
    document has "11.3 Performance assurance" against "12.3 Generic performance
    assurance management service"), which would leave the two stages of that
    service under different keys and break stage-2 retrieval for it.
    """
    parts = clause.split(".")
    return parts[1] if len(parts) > 1 else clause


def _umbrella_titles(sections: list[dict]) -> dict[str, str]:
    """Map each level-2 clause prefix to the key identifying its service.

    E.g. {"11.1": "1", "12.1": "1", "11.3": "3", "12.3": "3", ...}: the stage 2 and
    stage 3 blocks of one service share a key, so retrieval scoped to a service
    reaches both. A section is an umbrella when it is itself a level-2 clause.
    """
    umbrellas: dict[str, str] = {}
    for s in sections:
        clause = _clause_prefix(s["title"], levels=2)
        if clause and _clause_prefix(s["title"], levels=3) == clause:
            umbrellas.setdefault(clause, _service_key_from_clause(clause))
    return umbrellas


def umbrella_mns_for(section_title: str, sections: list[dict]) -> str:
    """Return the service key of the MnS that owns a section, or "" if unknown.

    Matches the `mns` metadata written by _label_target_spec_sections, so callers can build a
    retrieval filter that scopes to the same service as a given section.
    """
    clause2 = _clause_prefix(section_title, levels=2)
    return _umbrella_titles(sections).get(clause2, "")


def _label_target_spec_sections(sections: list[dict], stage3_clause: str = "") -> list[dict]:
    """Tag each section with its owning MnS and 3GPP stage, structurally.

    - mns: key of the service owning the section, taken from its level-2 umbrella
      clause (e.g. both "11.1.x" and "12.1.x" sections get key "1"). Derived purely
      from the clause numbering, so it works for any spec without hardcoded numbers.
    - stage: "3" when the section sits under the target's stage 3 block
      (stage3_clause), else "2". Best-effort; used only as a retrieval filter.

    Returns segment dicts ready for index_segments: {text, section_id, title,
    clause, mns, stage}.
    """
    # Level-2 umbrella titles keyed by their clause prefix (e.g. "11.1" -> title).
    umbrellas = _umbrella_titles(sections)

    segments = []
    for s in sections:
        clause2 = _clause_prefix(s["title"], levels=2)
        mns = umbrellas.get(clause2, "")
        in_stage3 = bool(stage3_clause) and (
            clause2 == stage3_clause
            or _clause_prefix(s["title"], levels=len(stage3_clause.split("."))) == stage3_clause
            or s["title"].strip().split(" ", 1)[0].startswith(stage3_clause + ".")
        )
        segments.append({
            "text":       f"{s['title']}\n{s['content']}",
            "section_id": s["section_id"],
            "title":      s["title"],
            "clause":     clause2,
            "mns":        mns,
            "stage":      "3" if in_stage3 else "2",
        })
    return segments


def index_target_spec_sections(sections: list[dict], stage3_clause: str = "") -> str:
    """Index the document's sections into a fresh temporary collection.

    Each chunk keeps mns/stage/clause metadata so retrieval can be scoped to one
    service and stage. Returns the collection name (unique per call).
    """
    collection_name = f"{SPEC_COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}"
    segments = _label_target_spec_sections(sections, stage3_clause=stage3_clause)
    index_segments(
        segments,
        factory=_spec_factory(),
        collection_name=collection_name,
        chunk_size=SPEC_CHUNK_SIZE,
        chunk_overlap=SPEC_CHUNK_OVERLAP,
        embedding_dim=EMBEDDING_DIM,
    )
    logger.info(f"Indexed {len(segments)} spec section(s) into '{collection_name}'.")
    return collection_name


def search_target_spec(
    query: str,
    collection_name: str,
    k: int = SPEC_RETRIEVE_CHUNKS,
    filters: dict | None = None,
) -> str:
    """Retrieve the most relevant spec chunks for a query, optionally filtered.

    Dense similarity search over the temporary spec collection. Pass filters
    (e.g. {"mns": "...", "stage": "2"}) to scope retrieval to one service/stage.
    Returns the chunk texts joined for prompt injection, or "" if unavailable.
    """
    if not collection_name:
        return ""
    factory = _spec_factory()
    if not factory.client.collection_exists(collection_name):
        logger.warning(f"Spec collection '{collection_name}' not found — skipping spec RAG.")
        return ""
    result = retrieve(
        query, factory=factory, collection_name=collection_name, k=k, filters=filters,
    )
    if not result.docs:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in result.docs)


def drop_target_spec_collection(collection_name: str) -> None:
    """Remove the temporary spec collection at the end of a run."""
    if collection_name:
        drop_collection(_spec_factory(), collection_name)


def create_rules_collection() -> str:
    """Create an empty temporary collection for rules validated during this run.

    The Validator indexes each rule as it passes, then queries this collection to
    spot rules duplicating ones already accepted in other sections. Returns the
    collection name; the Builder drops it when the run ends.
    """
    collection_name = f"{RULES_COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}"
    client = _spec_factory().client
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
    )
    logger.info(f"Created temporary rules collection '{collection_name}'.")
    return collection_name


def _rule_text_for_index(rule: dict) -> str:
    """Render a rule as the compact text indexed for similarity search.

    Carries the fields that identify a rule — section, type, name, text and the
    OpenAPI mapping — so a duplicate can be recognised and traced back to where it
    came from. The reflection fields (reasoning, confidence, flags) and validation
    notes are left out: they are verbose and add no signal for duplicate detection.
    """
    m = rule.get("openapi_mapping", {}) or {}
    return (
        f"[{rule.get('section_id', '')}] {rule.get('section_title', '')}\n"
        f"[{rule.get('rule_type', '')}] {rule.get('source_name', '')}: "
        f"{rule.get('rule_text', '')} | "
        f"{m.get('openapi_object', '')} {m.get('openapi_field', '')} {m.get('openapi_value', '')}"
    )


def index_validated_rule(rule: dict, collection_name: str) -> None:
    """Index one validated rule so later sections can be checked against it.

    No-op when the collection is unset/missing, so the pipeline still runs when
    the rules RAG is unavailable.
    """
    if not collection_name:
        return
    factory = _spec_factory()
    if not factory.client.collection_exists(collection_name):
        logger.warning(f"Rules collection '{collection_name}' not found — skipping index.")
        return
    metadata = {
        "section_id":    rule.get("section_id", ""),
        "section_title": rule.get("section_title", ""),
        "rule_type":     rule.get("rule_type", ""),
        "source_name":   rule.get("source_name", ""),
    }
    vector_store = factory.get_vector_store(collection_name, sparse_vector_name=None)
    vector_store.add_texts(texts=[_rule_text_for_index(rule)], metadatas=[metadata])


def search_rules_by_text(
    query: str,
    collection_name: str,
    k: int = RULES_RETRIEVE_CHUNKS,
    exclude_section_id: str = "",
) -> str:
    """Retrieve already-validated rules most similar to a free-text query.

    The shared core behind rule-duplication checks. The Validator queries with one
    rule's text; the Extractor queries with a whole section's text to see which of
    its rules already exist elsewhere. Rules from `exclude_section_id` are dropped
    (the same-section case is handled separately). Returns the compact rule texts
    joined for prompt injection, or "" when the collection is unset/missing.
    """
    if not collection_name:
        return ""
    factory = _spec_factory()
    if not factory.client.collection_exists(collection_name):
        return ""
    result = retrieve(
        query,
        factory=factory,
        collection_name=collection_name,
        k=k + 5,  # over-fetch so same-section hits can be filtered out
    )
    lines = [
        d.page_content
        for d in result.docs
        if d.metadata.get("section_id") != exclude_section_id
    ][:k]
    return "\n\n".join(lines)


def search_similar_rules(
    rule: dict,
    collection_name: str,
    k: int = RULES_RETRIEVE_CHUNKS,
    exclude_section_id: str = "",
) -> str:
    """Retrieve rules already validated that resemble the given rule.

    Used by the Validator to detect a rule duplicating one accepted in another
    section. Delegates to search_rules_by_text using the rule's compact text as
    the query. Returns the rules joined for prompt injection, or "" when none.
    """
    return search_rules_by_text(
        _rule_text_for_index(rule), collection_name, k=k, exclude_section_id=exclude_section_id,
    )


def drop_rules_collection(collection_name: str) -> None:
    """Remove the temporary rules collection at the end of a run."""
    if collection_name:
        drop_collection(_spec_factory(), collection_name)


def _resolve_corpus_collection(factory) -> str:
    """First configured corpus collection that actually exists in Qdrant, or "".

    GPP_CORPUS_COLLECTIONS lists candidates in preference order (a full-release
    index first, the 28-series subset as fallback), so the pipeline uses whatever
    was indexed without a config change. Returns "" when none is present.
    """
    for name in GPP_CORPUS_COLLECTIONS:
        if factory.client.collection_exists(name):
            return name
    return ""


def search_3gpp_corpus(
    query: str,
    k: int = GPP_CORPUS_RETRIEVE_CHUNKS,
    filters: dict | None = None,
) -> str:
    """Retrieve chunks from the persistent 3GPP corpus collection for a query.

    Resolves the corpus collection from GPP_CORPUS_COLLECTIONS (first candidate
    present in Qdrant). Used to ground rules in referenced 3GPP specs. Returns
    joined chunk texts, or "" (with a warning) when no candidate collection has
    been indexed — the pipeline then proceeds without corpus RAG.
    """
    factory = _spec_factory()
    collection = _resolve_corpus_collection(factory)
    if not collection:
        logger.warning(
            f"No 3GPP corpus collection found (tried {GPP_CORPUS_COLLECTIONS}) — "
            "run qdrant_3gpp_rag/scripts/index_3gpp_all.py to build one. "
            "Continuing without corpus RAG."
        )
        return ""
    result = retrieve(
        query, factory=factory, collection_name=collection, k=k, filters=filters,
    )
    if not result.docs:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in result.docs)


def search_imported_type_ref(query: str, k: int = GPP_CORPUS_RETRIEVE_CHUNKS) -> str:
    """Fetch a common type's definition from the corpus of other 3GPP specs.

    3GPP management-service specs reuse common types (e.g. NotificationHeader,
    ErrorResponse) defined in a shared spec like TS 28.623 and referenced via an
    `allOf $ref`. A section that embeds such a type lists the type's fields inline
    without saying which are inherited; to tell them apart we need the imported
    type's OWN definition. The query is supplied by the caller (the analyzer's
    step 1 writes it to match the definition block), and this returns the matching
    corpus chunks, or "" when the corpus is absent or nothing matches.
    """
    if not query:
        return ""
    return search_3gpp_corpus(query, k=k)


# ---------------------------------------------------------------------------
# Cached singletons — loaded once per process, reused across all calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    """Returns a cached Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFace embeddings model (loads model weights once)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store() -> QdrantVectorStore:
    """Returns a QdrantVectorStore bound to the OpenAPI reference collection."""
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=OPENAPI_COLLECTION_NAME,
        embedding=_get_embeddings(),
        vector_name="text-dense",
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def chunk_document(text: str) -> list[str]:
    """
    Splits the OpenAPI reference into chunks, respecting markdown section headers.

    Uses the shared chunk_markdown_by_headers helper so the reference (a
    header-structured spec) is split on '#'/'##'/'###' boundaries before size —
    keeping objects like "Parameter Object" whole instead of cut mid-section.
    Chunk size and overlap are configured in settings.py.
    """
    return chunk_markdown_by_headers(
        text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )


def index_openapi_reference(force: bool = False) -> None:
    """
    Indexes the local OpenAPI reference document into Qdrant.

    Skips indexing if the collection already has vectors (unless force=True).
    Run once after fetch_openapi_reference() to populate the RAG store.

    Args:
        force: If True, drops and recreates the collection before indexing.

    Raises:
        ValueError: If the OpenAPI reference file is missing or empty.
    """
    client = _get_client()

    # Guard: skip if already indexed
    if client.collection_exists(OPENAPI_COLLECTION_NAME) and not force:
        count = client.count(OPENAPI_COLLECTION_NAME).count
        if count > 0:
            logger.info(
                f"Collection '{OPENAPI_COLLECTION_NAME}' already has {count} vector(s). "
                "Skipping. Use force=True to reindex."
            )
            return

    # Load reference document
    text = load_openapi_reference()
    if not text:
        raise ValueError(
            "OpenAPI reference is empty. Run fetch_openapi_reference() first."
        )

    # Chunk document
    chunks = chunk_document(text)
    logger.info(f"Chunked OpenAPI reference into {len(chunks)} chunk(s).")

    # (Re)create collection with named dense vector
    if client.collection_exists(OPENAPI_COLLECTION_NAME):
        client.delete_collection(OPENAPI_COLLECTION_NAME)
        logger.debug(f"Dropped existing collection '{OPENAPI_COLLECTION_NAME}'.")

    client.create_collection(
        collection_name=OPENAPI_COLLECTION_NAME,
        vectors_config={
            "text-dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
    )
    logger.info(
        f"Collection '{OPENAPI_COLLECTION_NAME}' created "
        f"(model={EMBEDDING_MODEL}, dim={EMBEDDING_DIM})."
    )

    # Embed and ingest via LangChain vector store
    vector_store = _get_vector_store()
    vector_store.add_texts(texts=chunks)

    total = client.count(OPENAPI_COLLECTION_NAME).count
    logger.info(f"Indexing complete — {total} vector(s) in '{OPENAPI_COLLECTION_NAME}'.")


def search_openapi_reference(query: str, k: int = OPENAPI_REFERENCE_RETRIEVE_CHUNKS) -> str:
    """
    Searches the OpenAPI reference collection in Qdrant for the top-k most
    relevant chunks matching the query.

    Returns chunk texts joined by separators, ready to inject into a prompt.
    Returns "" with a warning if the collection is missing or empty.

    Args:
        query: Natural language query (e.g. section title + extraction focus).
        k:     Number of chunks to retrieve (default: 5).
    """
    client = _get_client()

    if not client.collection_exists(OPENAPI_COLLECTION_NAME):
        logger.warning(
            f"Collection '{OPENAPI_COLLECTION_NAME}' not found. "
            "Run index_openapi_reference() first."
        )
        return ""

    vector_store = _get_vector_store()
    docs = vector_store.similarity_search(query, k=k)

    if not docs:
        return ""

    texts = [doc.page_content for doc in docs]
    logger.debug(f"Retrieved {len(texts)} chunk(s) for: '{query[:60]}'")
    return "\n\n---\n\n".join(texts)
