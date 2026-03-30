# src/rag/retriever.py
"""
Semantic retrieval from Qdrant for the Reflector Node.

The Reflector calls get_relevant_chunks(rule_text) to retrieve 3GPP spec
chunks that support or contradict each extracted rule before CoT reasoning.

Two retrieval modes:
  - filters provided  → exact metadata filter + vector similarity
  - no filters        → SelfQueryRetriever (LLM parses query into filter + search)

Ported from openapi_chatbotUI — only import paths changed.
"""

from typing import Dict, List, Optional, Any

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import get_logger
from config.llm_config import llm
from config.hardware import device
from rag.qdrant_factory import QdrantFactory
from rag.self_query import metadata_field_info, document_content_description

logger = get_logger(__name__)


def _build_self_query_retriever(k: int = 5) -> SelfQueryRetriever:
    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        search_kwargs={"k": k},
    )


def _format_doc(doc: Document) -> str:
    """Formats a Document as string with metadata header + content."""
    md = doc.metadata or {}
    header = (
        f"release:     {md.get('release', 'unknown')}\n"
        f"series:      {md.get('series', 'unknown')}\n"
        f"spec:        {md.get('spec', 'unknown')}\n"
        f"chunk_index: {md.get('chunk_index', 'unknown')}\n\n"
    )
    return header + doc.page_content.strip()


def retrieve(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Core retrieval — returns List[Document].

    filters example: {"release": "Rel-18", "spec": "28532"}
    """
    logger.info(f"Retrieving | query={query!r} | k={k} | filters={filters}")

    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()

    if filters:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )
        docs = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)
    else:
        retriever = _build_self_query_retriever(k=k)
        docs = retriever.invoke(query)

    logger.info(f"Retrieved {len(docs)} chunks")
    return docs


def get_relevant_chunks(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Main entry point for the Reflector Node.

    Returns a list of formatted strings (metadata + content) ready to be
    injected into the LLM prompt for CoT grounding.
    """
    docs = retrieve(query=query, k=k, filters=filters)
    return [_format_doc(doc) for doc in docs]
