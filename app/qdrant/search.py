"""Qdrant search operations."""

from qdrant_client import models

from app.core.config import settings
from app.core.logging import logger
from app.qdrant.client import qdrant_client


def search_profiles(
    query_vector: list[float],
    limit: int = 20,
    filters: dict | None = None,
) -> list:
    """Search candidate profile vectors."""
    client = qdrant_client.client

    must_conditions = []
    if filters:
        if filters.get("experience_min") is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="experience",
                    range=models.Range(gte=filters["experience_min"]),
                )
            )
        if filters.get("experience_max") is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="experience",
                    range=models.Range(lte=filters["experience_max"]),
                )
            )
        if filters.get("skills"):
            for skill in filters["skills"]:
                must_conditions.append(
                    models.FieldCondition(
                        key="skills",
                        match=models.MatchValue(value=skill.lower()),
                    )
                )
        if filters.get("location"):
            must_conditions.append(
                models.FieldCondition(
                    key="location",
                    match=models.MatchValue(value=filters["location"]),
                )
            )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=settings.PROFILE_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    ).points

    logger.info(f"Profile search returned {len(results)} results")
    return results


def search_interviews(
    query_vector: list[float],
    limit: int = 50,
    candidate_id: str | None = None,
) -> list:
    """Search interview chunk vectors."""
    client = qdrant_client.client

    must_conditions = []
    if candidate_id:
        must_conditions.append(
            models.FieldCondition(
                key="candidate_id",
                match=models.MatchValue(value=candidate_id),
            )
        )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=settings.INTERVIEW_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    ).points

    logger.info(f"Interview search returned {len(results)} results")
    return results


def get_candidate_interviews(candidate_id: str, limit: int = 100) -> list:
    """Get all interview chunks for a candidate."""
    client = qdrant_client.client

    results = client.scroll(
        collection_name=settings.INTERVIEW_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="candidate_id",
                    match=models.MatchValue(value=candidate_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return results


def get_all_candidates(limit: int = 100) -> list:
    """Get all candidate profiles."""
    client = qdrant_client.client

    results = client.scroll(
        collection_name=settings.PROFILE_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return results
