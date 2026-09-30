"""Profile retrieval from Qdrant vector database."""

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.service import embedding_service
from app.qdrant.search import search_profiles
from app.schemas.search import InterpretedRequirements


def retrieve_profiles(
    query: str,
    requirements: InterpretedRequirements,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve candidate profiles matching the query and requirements.

    Embeds the semantic query, builds structured filters from the
    interpreted requirements, and searches the profile collection in Qdrant.

    Args:
        query: Natural language search query.
        requirements: Structured requirements parsed from the query.
        limit: Maximum number of results. Defaults to ``settings.TOP_K_PROFILE``.

    Returns:
        List of dicts, each containing ``candidate_id``, ``score``, and
        ``payload`` keys with the raw Qdrant payload data.
    """
    if limit is None:
        limit = settings.TOP_K_PROFILE

    # Embed the semantic query into a vector
    query_vector = embedding_service.encode(query)

    # Build structured filters from requirements
    # Only use experience as a hard filter - skills and location are soft preferences
    filters: dict = {}
    if requirements.experience_min is not None:
        filters["experience_min"] = requirements.experience_min
    if requirements.experience_max is not None:
        filters["experience_max"] = requirements.experience_max

    # Execute the vector search
    results = search_profiles(
        query_vector=query_vector,
        limit=limit,
        filters=filters if filters else None,
    )

    # Format results into plain dicts
    formatted: list[dict] = []
    for r in results:
        formatted.append({
            "candidate_id": r.payload.get("candidate_id", ""),
            "score": r.score,
            "payload": r.payload,
        })

    logger.info(
        "Retrieved %d profiles for query: %s...",
        len(formatted),
        query[:50],
    )
    return formatted
