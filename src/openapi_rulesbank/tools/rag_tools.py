# src/tools/rag_tools.py

"""
RAG tools for indexing and retrieving OpenAPI reference chunks from Qdrant.

Every collection goes through qdrant_3gpp_rag's QdrantFactory, the one model and
vector layout shared by all repositories (dense "text-dense" + BM25 "text-sparse",
searched hybrid when the collection has the sparse vector).

The Qdrant collection is created once (via index_openapi_reference()) and
persists across pipeline runs. Extractor and Reflector nodes call
retrieve_chunks() directly — no re-indexing needed.

Functions:
    chunk_document(text)           — cuts the reference along its heading tree
    index_openapi_reference(force) — indexes openapi_reference.md into Qdrant
    search_openapi_reference(query, k) — semantic search, returns joined chunk texts
"""

import os
import re
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Distance, VectorParams

from qdrant_3gpp_rag import (
    QdrantFactory,
    chunk_markdown_tree,
    chunk_spec_sections,
    index_segments,
    drop_collection,
    retrieve,
)

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.hardware import device
from openapi_rulesbank.config.settings import (
    QDRANT_HOST, QDRANT_PORT,
    OPENAPI_COLLECTION_NAME,
    OPENAPI_REFERENCE_VERSION,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    OPENAPI_REFERENCE_RETRIEVE_CHUNKS,
    SPEC_COLLECTION_PREFIX, SPEC_RETRIEVE_CHUNKS,
    SPEC_CHUNK_SIZE, SPEC_CHUNK_OVERLAP,
    GPP_CORPUS_COLLECTIONS, GPP_CORPUS_RETRIEVE_CHUNKS,
    EXCLUDE_TARGET_SPEC_FROM_CORPUS, EXCLUDE_TARGET_ANNEX,
    IMPORTED_TYPE_RETRIEVE_CHUNKS,
    RULES_COLLECTION_PREFIX, RULES_RETRIEVE_CHUNKS,
)
from openapi_rulesbank.tools.document_tools import load_openapi_reference

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Spec RAG — index the main 3GPP document into a temporary Qdrant collection
# and retrieve related sections during extraction. Backed by qdrant_3gpp_rag.
# ---------------------------------------------------------------------------

def _spec_factory() -> QdrantFactory:
    """The shared QdrantFactory, for every collection this project reads or builds."""
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


def _label_spec_chunks(chunks: list[dict], sections: list[dict],
                       stage3_clause: str = "") -> list[dict]:
    """Add our mns/stage retrieval filters to chunks cut by chunk_spec_sections.

    The chunks carry the corpus format (heading path, clause, defines, mentions,
    annex, part/parts); mns and stage are computed from each chunk's own clause
    number, the same way _label_target_spec_sections does from a section title.
    """
    umbrellas = _umbrella_titles(sections)
    for c in chunks:
        clause = c.get("clause", "") or ""
        clause2 = ".".join(clause.split(".")[:2]) if clause[:1].isdigit() else ""
        in_stage3 = bool(stage3_clause) and (
            clause == stage3_clause or clause.startswith(stage3_clause + ".")
        )
        c["mns"] = umbrellas.get(clause2, "")
        c["stage"] = "3" if in_stage3 else "2"
    return chunks


def index_target_spec_sections(
    sections: list[dict], stage3_clause: str = "", doc_text: str = "",
) -> str:
    """Index the document being processed into a fresh temporary collection.

    With doc_text, the document is cut by qdrant_3gpp_rag's chunk_spec_sections —
    the same format as the corpus (heading path at the start of each chunk;
    clause, title, path, defines, mentions, annex, part/parts in metadata) — and
    its annexes are left out when EXCLUDE_TARGET_ANNEX is on (the annex of this
    spec is the YAML being built). Without doc_text, the parsed sections are
    indexed as before. Every chunk keeps mns/stage so retrieval can be scoped to
    one service and stage. Returns the collection name (unique per call).
    """
    collection_name = f"{SPEC_COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}"
    if doc_text:
        segments = _label_spec_chunks(
            chunk_spec_sections(doc_text, include_annexes=not EXCLUDE_TARGET_ANNEX),
            sections, stage3_clause,
        )
        chunk_size, chunk_overlap = None, 0
    else:
        segments = _label_target_spec_sections(sections, stage3_clause=stage3_clause)
        chunk_size, chunk_overlap = SPEC_CHUNK_SIZE, SPEC_CHUNK_OVERLAP
    index_segments(
        segments,
        factory=_spec_factory(),
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dim=EMBEDDING_DIM,
    )
    logger.info(f"Indexed {len(segments)} spec chunk(s) into '{collection_name}'.")
    return collection_name


def search_target_spec(
    query: str,
    collection_name: str,
    k: int = SPEC_RETRIEVE_CHUNKS,
    filters: dict | None = None,
    exclude: list[dict] | None = None,
) -> str:
    """Retrieve the most relevant spec chunks for a query, optionally filtered.

    Dense similarity search over the temporary spec collection. Pass filters
    (e.g. {"mns": "...", "stage": "2"}) to scope retrieval to one service/stage,
    and exclude (e.g. [{"clause": "..."}]) to leave chunks out by metadata.
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
        exclude=exclude,
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


def target_spec_number(doc_path: str) -> str:
    """The spec number of the document being processed, as the corpus records it.

    "…/28532-i20.md" -> "28532"; "…/38101-1-i20.md" -> "38101-1". Matches the
    corpus metadata `spec_number` (dots removed), so the target spec can be left
    out of corpus searches. "" when the file name carries no spec number.
    """
    stem = os.path.splitext(os.path.basename(doc_path or ""))[0]
    m = re.match(r"^(\d{3,5}(?:-\d+)?)-[a-z0-9]{3}", stem)
    return m.group(1) if m else ""


def _exclude_target_spec(exclude_spec: str) -> list[dict] | None:
    """The exclusion group that keeps the spec being processed out of the corpus.

    The whole spec, any version — its content is searched in its own collection,
    indexed at run time from the exact version processed. Every other spec stays
    searchable. None when the flag is off or the spec is unknown.
    """
    if EXCLUDE_TARGET_SPEC_FROM_CORPUS and exclude_spec:
        return [{"spec_number": exclude_spec}]
    return None


def _corpus_docs(
    query: str,
    k: int,
    filters: dict | None = None,
    exclude_spec: str = "",
) -> list:
    """Corpus documents (text + metadata) for a query, or [] without a corpus."""
    factory = _spec_factory()
    collection = _resolve_corpus_collection(factory)
    if not collection:
        logger.warning(
            f"No 3GPP corpus collection found (tried {GPP_CORPUS_COLLECTIONS}) — "
            "run qdrant_3gpp_rag/scripts/index_3gpp_all.py to build one. "
            "Continuing without corpus RAG."
        )
        return []
    result = retrieve(
        query, factory=factory, collection_name=collection, k=k, filters=filters,
        exclude=_exclude_target_spec(exclude_spec),
    )
    return list(result.docs or [])


def search_3gpp_corpus(
    query: str,
    k: int = GPP_CORPUS_RETRIEVE_CHUNKS,
    filters: dict | None = None,
    exclude_spec: str = "",
) -> str:
    """Retrieve chunks from the persistent 3GPP corpus collection for a query.

    Resolves the corpus collection from GPP_CORPUS_COLLECTIONS (first candidate
    present in Qdrant). Used to ground rules in referenced 3GPP specs. exclude_spec
    is the spec number of the document being processed: that spec is left out
    (EXCLUDE_TARGET_SPEC_FROM_CORPUS), its content coming from its own collection.
    Returns joined chunk
    texts, or "" when nothing comes back or no candidate collection exists — the
    pipeline then proceeds without corpus RAG.
    """
    docs = _corpus_docs(query, k=k, filters=filters, exclude_spec=exclude_spec)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def spec_number_of(spec_ref: str) -> str:
    """The corpus spec_number of a spec as the text cites it, or "".

    "TS 28.623 [44]" -> "28623"; "TS 38.101-1" -> "38101-1". Same form as the
    corpus metadata `spec_number` (dots removed).
    """
    m = re.search(r"(\d{2})\.(\d{3})(-\d+)?", spec_ref or "")
    return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}" if m else ""


def search_cited_spec(subject: str, cited_spec: str, exclude_spec: str = "") -> str:
    """Corpus chunks about a subject, from the spec the clause cites for it.

    Scoped to the cited spec's spec_number, so another spec on the same subject
    does not answer for it; unscoped when the citation carries no number or the
    cited spec is not in the corpus. "" when the cited spec is the one being
    processed: it is left out of the corpus, and its text is in its own collection.
    """
    number = spec_number_of(cited_spec)
    if number and number == exclude_spec:
        return ""
    if number:
        found = search_3gpp_corpus(subject, filters={"spec_number": number},
                                   exclude_spec=exclude_spec)
        if found:
            return found
    return search_3gpp_corpus(f"{subject} {cited_spec}", exclude_spec=exclude_spec)


def search_imported_type_ref(
    type_name: str,
    key_terms: str = "",
    k: int = IMPORTED_TYPE_RETRIEVE_CHUNKS,
    exclude_spec: str = "",
) -> str:
    """Fetch a common type's definition from the corpus of other 3GPP specs.

    3GPP management-service specs reuse common types (e.g. NotificationHeader,
    ErrorResponse) defined in a shared spec like TS 28.623 and referenced via an
    `allOf $ref`. To tell a section's own fields from the inherited ones we need
    the imported type's OWN definition block.

    The corpus records, per chunk, the types it defines (metadata `defines`), so
    the definition is first asked for exactly: the chunks that define this type.
    Only when none does (a type the corpus does not mark) the earlier similarity
    query runs — the type name plus the shape of an OpenAPI schema definition
    ("type: object properties required"), seeded with the analyzer's key_terms.
    Returns the matching chunks, or "" when the type is unknown or the corpus is
    absent.
    """
    if not type_name:
        return ""
    query = f"{type_name} type: object properties required {key_terms}".strip()
    defining = _corpus_docs(query, k=k, filters={"defines": type_name},
                            exclude_spec=exclude_spec)
    if defining:
        return "\n\n---\n\n".join(doc.page_content for doc in defining)
    return search_3gpp_corpus(query, k=k, exclude_spec=exclude_spec)


# A 3GPP OpenAPI file reference: "<File>.yaml#/components/schemas/<TypeName>".
# Captures the filename so, given a type, the file it is published in can be read
# back from text (helper_context, annex, or a retrieved corpus chunk). Pandoc
# escapes '_' and '#' as '\_' and '\#' in the annex Markdown, so a backslash is
# allowed (and stripped) before each such character; the filename and type names
# are read with the escapes removed.
_YAML_REF = re.compile(
    r"((?:[A-Za-z0-9]|\\?_)+\.yaml)\s*\\?#\s*/\s*components\s*/\s*schemas\s*/\s*"
    r"((?:[A-Za-z0-9]|\\?_)+)"
)


def _unescape(s: str) -> str:
    """Drop pandoc's backslash escapes before '_' and '#'."""
    return s.replace("\\_", "_").replace("\\#", "#")


def _files_for_type_in_text(type_name: str, text: str) -> list[str]:
    """Every YAML file a block of text ties to `type_name`, as a sorted list.

    Collects all '<File>.yaml#/components/schemas/<type_name>' references and
    returns the distinct file parts. Deterministic reading — it never derives a
    name from a spec number, only reports the ones the text writes out. Pandoc's
    '\\_'/'\\#' escapes are normalised. Returns [] when the text ties the type to
    no file, one entry when unambiguous, several when the text itself maps the
    same type to more than one file (e.g. 'resources', published by each NRM
    under its own file) — the caller then lets the LLM pick, never guesses.
    """
    if not type_name or not text:
        return []
    files = {
        _unescape(m.group(1))
        for m in _YAML_REF.finditer(text)
        if _unescape(m.group(2)) == type_name
    }
    return sorted(files)


def _files_from_defining_chunks(type_name: str, exclude_spec: str = "") -> list[str]:
    """YAML files that publish `type_name`, read from the corpus metadata.

    A chunk whose metadata says it DEFINES the type and that sits in an annex
    carries the OpenAPI document it belongs to in its title (e.g. 'C.4.2a OpenAPI
    document "TS28623_ComDefs.yaml"'). Each such title gives one file. Read only,
    never built; the spec being processed is excluded like in every corpus search.
    """
    docs = _corpus_docs(f"{type_name} OpenAPI document components schemas", k=20,
                        filters={"defines": type_name}, exclude_spec=exclude_spec)
    files = set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        meta = meta.get("metadata", meta)          # langchain may nest it once more
        if not meta.get("annex"):
            continue
        title = _unescape((meta.get("title") or "").replace('\\"', '"'))
        files.update(re.findall(r"[A-Za-z0-9_]+\.yaml", title))
    return sorted(files)


def _prefer_source_spec(options: list[str], source_spec: str) -> list[str]:
    """Narrow several candidate files to those of the type's own spec, if any.

    3GPP names a spec's OpenAPI files "TS<number>_<…>.yaml", so when the spec a
    type is imported from is known (e.g. "TS 28.623"), the files of that spec are
    the ones that can hold it. Keeps the full list when none matches, and changes
    nothing for a single option.
    """
    digits = re.sub(r"\D", "", source_spec or "")
    if len(options) < 2 or not digits:
        return options
    own = [f for f in options if f.startswith(f"TS{digits}_")]
    return own or options


def resolve_external_filename(
    type_name: str,
    source_spec: str = "",
    helper_context: str = "",
    annex_text: str = "",
    exclude_spec: str = "",
) -> tuple[str, list[str], str]:
    """Resolve the YAML filename an external $ref to `type_name` must carry.

    A cascade, stopping at the first source that offers any filename, in priority
    order:
      1. helper_context — whatever the USER supplied (may be empty); highest
         authority, an explicit human-provided override.
      2. annex_text — the document's own embedded OpenAPI annex, read ONLY for
         the filenames it spells out; skipped when the document has none.
      3. corpus RAG — first the chunks the corpus marks as DEFINING the type
         (their annex title names the file), then the text of a query scoped to
         the known source_spec, then a broader one;
      4. only when source_spec is known: the naming convention,
         "TS<spec digits>_ComDefs.yaml", source "convention-fallback" — built, not
         read, so the rule using it is scored at the acceptance floor.
    Each source is read into the FULL set of files it ties to the type; when the
    source spec is known, files of other specs are dropped (_prefer_source_spec).
    Returns (filename, options, source):
      - exactly one file at the first source that answers -> (file, [file], src):
        unambiguous, the caller uses it directly;
      - several files -> ("", options, src + ":ambiguous"): the caller hands the
        options to the LLM, which picks by the using clause's context;
      - nothing anywhere -> ("", [], "unresolved"): the caller defers / declares a
        gap and NEVER invents a name.
    src is "helper_context" | "annex" | "rag:defines" | "rag:source-spec" |
    "rag:broad". exclude_spec keeps the spec being processed out of the corpus
    searches. Corpus searches run only when the sources before them gave nothing.
    """
    if not type_name:
        return "", [], "unresolved"

    sources = [
        ("helper_context", lambda: _files_for_type_in_text(type_name, helper_context)),
        ("annex",          lambda: _files_for_type_in_text(type_name, annex_text)),
        ("rag:defines",    lambda: _files_from_defining_chunks(type_name, exclude_spec)),
    ]
    if source_spec:
        sources.append(("rag:source-spec", lambda: _files_for_type_in_text(
            type_name, search_3gpp_corpus(
                f"{source_spec} {type_name} OpenAPI file name components schemas .yaml",
                exclude_spec=exclude_spec))))
    sources.append(("rag:broad", lambda: _files_for_type_in_text(
        type_name, search_3gpp_corpus(
            f"{type_name} OpenAPI file name components schemas .yaml",
            exclude_spec=exclude_spec))))

    for src, read in sources:
        options = _prefer_source_spec(read(), source_spec)
        if len(options) == 1:
            return options[0], options, src
        if len(options) > 1:
            return "", options, src + ":ambiguous"

    # 4. Last resort, only when the spec is known: the file-name convention
    # ("TSxxyyy_<group>.yaml"), with the common-definitions group of the 28.x
    # family, "ComDefs". This one is BUILT, not read — so the source says so and
    # the Reflector scores the rule at the acceptance floor and explains it.
    digits = re.sub(r"\D", "", source_spec or "")
    if digits:
        fallback = f"TS{digits}_ComDefs.yaml"
        return fallback, [fallback], "convention-fallback"

    return "", [], "unresolved"


def _get_vector_store():
    """The OpenAPI reference collection, through the shared factory."""
    return _spec_factory().get_vector_store(OPENAPI_COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def chunk_document(text: str) -> list[dict]:
    """
    Cuts the OpenAPI reference along its own heading tree (qdrant_3gpp_rag's
    chunk_markdown_tree): a section that fits is one chunk, a larger one is cut at
    its child headings, small pieces join a sibling, and every chunk starts with its
    heading path ("... › Parameter Object › Fixed Fields (part 2/4)"). The object
    level differs by version (3.0.1 "####", 3.2.0 "###"), so no level is fixed.
    Returns segments (text + title / path / part / parts) for index_segments.
    """
    return chunk_markdown_tree(text, {"openapi_version": OPENAPI_REFERENCE_VERSION})


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
    client = _spec_factory().client

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
    segments = chunk_document(text)
    logger.info(f"Cut OpenAPI reference {OPENAPI_REFERENCE_VERSION} into {len(segments)} chunk(s).")

    # (Re)create the collection (dense + BM25) and index the chunks as they are
    index_segments(
        segments,
        factory=_spec_factory(),
        collection_name=OPENAPI_COLLECTION_NAME,
        chunk_size=None,
        embedding_dim=EMBEDDING_DIM,
    )
    logger.info(
        f"Collection '{OPENAPI_COLLECTION_NAME}' rebuilt "
        f"(model={EMBEDDING_MODEL}, dim={EMBEDDING_DIM})."
    )

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
    client = _spec_factory().client

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
