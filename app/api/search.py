"""Search API endpoints."""

from fastapi import APIRouter, HTTPException

from app.core.logging import logger
from app.llm.query_parser import parse_query
from app.retrieval.candidate_ranker import search_pipeline
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for candidates using natural language query.

    Parses the query, retrieves profile and interview evidence,
    ranks candidates, and returns results with AI summaries.
    """
    logger.info(f"Search request: {request.query[:100]}")

    try:
        # Parse natural language query
        requirements = parse_query(request.query)

        # Execute search pipeline
        results = search_pipeline(request.query, requirements)

        # Limit results
        results = results[: request.top_k]

        return SearchResponse(
            query=request.query,
            interpreted_requirements=requirements,
            results=results,
            result_count=len(results),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/api/search")
async def search_get(q: str, top_k: int = 10):
    """GET endpoint for search (convenience)."""
    return await search(SearchRequest(query=q, top_k=top_k))
