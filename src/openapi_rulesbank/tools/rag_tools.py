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

from qdrant_3gpp_rag import QdrantFactory, index_segments, drop_collection, retrieve

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.hardware import device
from openapi_rulesbank.config.settings import (
    QDRANT_HOST, QDRANT_PORT,
    OPENAPI_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    OPENAPI_REFERENCE_RETRIEVE_CHUNKS,
    SPEC_COLLECTION_PREFIX, SPEC_RETRIEVE_CHUNKS,
    GPP_CORPUS_COLLECTION, GPP_CORPUS_RETRIEVE_CHUNKS,
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


def _service_name_from_title(title: str) -> str:
    """Strip the leading clause number from a title, leaving the service name.

    "11.1 Generic provisioning management service" -> "Generic provisioning
    management service". This name is stable across the stage 2 (e.g. 11.1) and
    stage 3 (e.g. 12.1) copies of the same service, so it groups them together.
    """
    parts = title.strip().split(" ", 1)
    return parts[1].strip() if len(parts) > 1 else title.strip()


def _umbrella_titles(sections: list[dict]) -> dict[str, str]:
    """Map each level-2 clause prefix to its service name (number stripped).

    E.g. {"11.1": "Generic provisioning management service",
          "12.1": "Generic provisioning management service", ...}. A section is an
    umbrella when it is itself exactly a level-2 clause. Using the number-stripped
    name means the stage 2 and stage 3 blocks of one service share the same `mns`.
    """
    umbrellas: dict[str, str] = {}
    for s in sections:
        clause = _clause_prefix(s["title"], levels=2)
        if clause and _clause_prefix(s["title"], levels=3) == clause:
            umbrellas.setdefault(clause, _service_name_from_title(s["title"]))
    return umbrellas


def umbrella_mns_for(section_title: str, sections: list[dict]) -> str:
    """Return the MnS (umbrella title) that owns a section, or "" if unknown.

    Matches the `mns` metadata written by _label_sections, so callers can build a
    retrieval filter that scopes to the same service as a given section.
    """
    clause2 = _clause_prefix(section_title, levels=2)
    return _umbrella_titles(sections).get(clause2, "")


def _label_sections(sections: list[dict], stage3_clause: str = "") -> list[dict]:
    """Tag each section with its owning MnS and 3GPP stage, structurally.

    - mns: the title of the level-2 umbrella clause the section falls under
      (e.g. "11.1 Generic provisioning management service"). Derived purely from
      the clause numbering, so it works for any spec without hardcoded numbers.
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


def index_spec_sections(sections: list[dict], stage3_clause: str = "") -> str:
    """Index the document's sections into a fresh temporary collection.

    Each chunk keeps mns/stage/clause metadata so retrieval can be scoped to one
    service and stage. Returns the collection name (unique per call).
    """
    collection_name = f"{SPEC_COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}"
    segments = _label_sections(sections, stage3_clause=stage3_clause)
    index_segments(
        segments,
        factory=_spec_factory(),
        collection_name=collection_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_dim=EMBEDDING_DIM,
    )
    logger.info(f"Indexed {len(segments)} spec section(s) into '{collection_name}'.")
    return collection_name


def search_spec(
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


def drop_spec_collection(collection_name: str) -> None:
    """Remove the temporary spec collection at the end of a run."""
    if collection_name:
        drop_collection(_spec_factory(), collection_name)


def search_3gpp_corpus(
    query: str,
    k: int = GPP_CORPUS_RETRIEVE_CHUNKS,
    filters: dict | None = None,
) -> str:
    """Retrieve chunks from the persistent 3GPP corpus collection for a query.

    Reads GPP_CORPUS_COLLECTION (built once by qdrant_3gpp_rag's index_3gpp_all
    script). Used by the ContractBuilder to ground the contract in referenced
    3GPP specs. Returns joined chunk texts, or "" (with a warning) if the corpus
    collection has not been indexed — the pipeline then proceeds without it.
    """
    factory = _spec_factory()
    if not factory.client.collection_exists(GPP_CORPUS_COLLECTION):
        logger.warning(
            f"3GPP corpus collection '{GPP_CORPUS_COLLECTION}' not found — "
            "run qdrant_3gpp_rag/scripts/index_3gpp_all.py to build it. "
            "Continuing without corpus RAG."
        )
        return ""
    result = retrieve(
        query, factory=factory, collection_name=GPP_CORPUS_COLLECTION, k=k, filters=filters,
    )
    if not result.docs:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in result.docs)


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
    Splits text into overlapping chunks using RecursiveCharacterTextSplitter.
    Chunk size and overlap are configured in settings.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


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
