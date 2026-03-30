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

from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import get_logger
from config.hardware import device
from config.settings import (
    QDRANT_HOST, QDRANT_PORT,
    OPENAPI_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    OPENAPI_REFERENCE_RETRIEVE_CHUNKS,
)
from tools.document_tools import load_openapi_reference

logger = get_logger(__name__)


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
