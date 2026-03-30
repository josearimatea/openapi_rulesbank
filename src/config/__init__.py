# src/config/__init__.py
"""
Lightweight config entry point.

Only exposes get_logger — zero heavy dependencies (torch, langchain, etc.).
Import heavier pieces directly from their modules:
    from config.settings import QDRANT_HOST, OPENAI_API_KEY, ...
    from config.llm_config import llm
    from config.hardware import device
    from config.paths import OPENAPI_REFERENCE_DIR, ...
"""

_logging_configured = False


def get_logger(name: str):
    """
    Configures logging globally (if not already done) and returns
    a logger with the given name.

    Usage:
        logger = get_logger(__name__)
    """
    global _logging_configured
    if not _logging_configured:
        from . import logging_config  # noqa: F401 — executes basicConfig on first call
        _logging_configured = True

    import logging
    return logging.getLogger(name)


__all__ = ["get_logger"]