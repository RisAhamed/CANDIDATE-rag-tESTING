"""Qdrant client singleton."""

from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import logger


class QdrantClientSingleton:
    """Singleton Qdrant client."""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
            self._client = QdrantClient(url=settings.QDRANT_URL)
            logger.info("Qdrant client connected")

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        return self._client

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


# Global instance
qdrant_client = QdrantClientSingleton()
