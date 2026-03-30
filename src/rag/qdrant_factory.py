# src/rag/qdrant_factory.py
"""
Factory for HuggingFace embeddings and QdrantVectorStore.
Ported from openapi_chatbotUI — only the import path changed.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.settings import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME


class QdrantFactory:
    def __init__(self, device: str = "cpu"):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            
            timeout=120,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_qdrant_vector_store(self, collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            vector_name="text-dense",
            sparse_vector_name="text-sparse",
        )
