"""Interview retrieval from Qdrant vector database."""

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.service import embedding_service
from app.qdrant.search import search_interviews
from app.schemas.search import InterpretedRequirements


def retrieve_interviews(
    query: str,
    requirements: InterpretedRequirements,
    candidate_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve interview chunks matching the query and requirements.

    Embeds the query (augmented with interview-focused skills when
    available) and searches the interview collection in Qdrant.  When
    ``candidate_ids`` is provided the search is scoped to those
    candidates; otherwise it searches across all interviews.

    Args:
        query: Natural language search query.
        requirements: Structured requirements parsed from the query.
        candidate_ids: Optional list of candidate IDs to restrict the search.
        limit: Maximum number of results. Defaults to ``settings.TOP_K_INTERVIEW``.

    Returns:
        List of dicts, each containing ``candidate_id``, ``score``, and
        ``payload`` keys with the raw Qdrant payload data.
    """
    if limit is None:
        limit = settings.TOP_K_INTERVIEW

    # Build an interview-focused query by appending required skills
    interview_query = query
    if requirements.interview_required_skills:
        skills_text = ", ".join(requirements.interview_required_skills)
        interview_query = f"{query} Skills: {skills_text}"

    # Embed the query into a vector
    query_vector = embedding_service.encode(interview_query)

    # Execute the vector search
    all_results = []
    if candidate_ids:
        # Distribute the limit across candidates
        per_candidate_limit = max(1, limit // len(candidate_ids))
        for cid in candidate_ids:
            results = search_interviews(
                query_vector=query_vector,
                limit=per_candidate_limit,
                candidate_id=cid,
            )
            all_results.extend(results)
    else:
        all_results = search_interviews(
            query_vector=query_vector,
            limit=limit,
        )

    # Format results into plain dicts
    formatted: list[dict] = []
    for r in all_results:
        formatted.append({
            "candidate_id": r.payload.get("candidate_id", ""),
            "score": r.score,
            "payload": r.payload,
        })

    logger.info(
        "Retrieved %d interview chunks for query: %s...",
        len(formatted),
        query[:50],
    )
    return formatted
