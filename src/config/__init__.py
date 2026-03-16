# src/app/config/__init__.py
"""
Central configuration module.
Imports and exposes all config components for easy access.
Usage: from app.config import llm, device, QDRANT_HOST, etc.
"""

from .settings import (
    APP_ENV,
    TSPEC_DATA,
    CHUNKS_FILE,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    NUMBER_RETRIEVE_CHUNKS,
    OPENAI_API_KEY,
)
from .hardware import device
from .llm_config import llm
from .logging_config import log_level

_logging_configured = False

def get_logger(name):
    """
    Configures logging globally (if not already done) and returns
    a logger with the given name.
    
    Use this in notebooks and scripts:
        logger = get_logger(__name__)
    
    The name parameter allows module-specific logging (e.g., __name__).
    """
    global _logging_configured
    if not _logging_configured:
        from . import logging_config  # executes basicConfig, handlers, suppress httpx/httpcore
        _logging_configured = True
    
    import logging
    return logging.getLogger(name)

# # Error using import QdrantFactory in settings.py due to inexistent Collections at statup, moved to functions that need it
# # Global QdrantFactory instance (initialized after device)  
# from app.ingest.qdrant_factory import QdrantFactory
# factory = QdrantFactory(device=device)
# vector_store = factory.get_qdrant_vector_store()  # Pre-initialize for efficiency

__all__ = [
    "APP_ENV",
    "TSPEC_DATA",
    "CHUNKS_FILE",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "COLLECTION_NAME",
    "NUMBER_RETRIEVE_CHUNKS",
    "OPENAI_API_KEY",
    "device",
    "llm",
    "factory",
    "vector_store",
    "log_level",
]