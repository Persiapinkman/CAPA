from functools import lru_cache

from src.rag.service import RAGService


@lru_cache()
def get_rag_service() -> RAGService:
    """Return a shared RAGService instance for in-process API usage."""
    return RAGService()
