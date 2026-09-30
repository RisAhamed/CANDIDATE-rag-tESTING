"""Qdrant collection management."""

from qdrant_client import models

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.service import embedding_service
from app.qdrant.client import qdrant_client


def create_collections():
    """Create required Qdrant collections if they don't exist."""
    client = qdrant_client.client
    existing = [c.name for c in client.get_collections().collections]

    dimension = embedding_service.dimension

    # Profile collection
    if settings.PROFILE_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.PROFILE_COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.PROFILE_COLLECTION}")
    else:
        logger.info(f"Collection exists: {settings.PROFILE_COLLECTION}")

    # Interview collection
    if settings.INTERVIEW_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.INTERVIEW_COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.INTERVIEW_COLLECTION}")
    else:
        logger.info(f"Collection exists: {settings.INTERVIEW_COLLECTION}")


def get_collection_stats() -> dict:
    """Get statistics for all collections."""
    client = qdrant_client.client
    stats = {}

    for name in [settings.PROFILE_COLLECTION, settings.INTERVIEW_COLLECTION]:
        try:
            info = client.get_collection(collection_name=name)
            stats[name] = {
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            stats[name] = {"error": str(e)}

    return stats


def delete_collections():
    """Delete all collections (use with caution)."""
    client = qdrant_client.client
    for name in [settings.PROFILE_COLLECTION, settings.INTERVIEW_COLLECTION]:
        try:
            client.delete_collection(collection_name=name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.warning(f"Failed to delete {name}: {e}")
