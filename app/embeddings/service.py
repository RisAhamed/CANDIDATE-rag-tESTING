"""Embedding service using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    """Singleton embedding service."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def encode(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into embedding vectors."""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a search query."""
        return self.encode(query)


# Global instance
embedding_service = EmbeddingService()
