"""Candidate management API endpoints."""

from fastapi import APIRouter, HTTPException

from app.core.logging import logger
from app.qdrant.search import get_all_candidates, get_candidate_interviews
from app.schemas.candidate import CandidateResponse

router = APIRouter()


@router.get("/api/candidates")
async def list_candidates(limit: int = 100):
    """List all candidates in the database."""
    results = get_all_candidates(limit=limit)

    candidates = []
    if results and results[0]:
        seen_ids = set()
        for point in results[0]:
            payload = point.payload
            cid = payload.get("candidate_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(
                    CandidateResponse(
                        candidate_id=cid,
                        name=payload.get("candidate_name", ""),
                        email=payload.get("email", ""),
                        phone=payload.get("phone", ""),
                        location=payload.get("location", ""),
                        current_title=payload.get("title", ""),
                        total_experience=payload.get("experience", 0.0),
                        skills=payload.get("skills", []),
                        education=payload.get("education", ""),
                        companies=payload.get("companies", []),
                        profile_source=payload.get("document_id", ""),
                    )
                )

    return {"candidates": candidates, "count": len(candidates)}


@router.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Get detailed information about a specific candidate."""
    # Get profile
    profile_results = get_all_candidates(limit=1000)
    candidate_profile = None

    if profile_results and profile_results[0]:
        for point in profile_results[0]:
            payload = point.payload
            if payload.get("candidate_id") == candidate_id:
                candidate_profile = {
                    "candidate_id": candidate_id,
                    "name": payload.get("candidate_name", ""),
                    "email": payload.get("email", ""),
                    "phone": payload.get("phone", ""),
                    "location": payload.get("location", ""),
                    "current_title": payload.get("title", ""),
                    "total_experience": payload.get("experience", 0.0),
                    "skills": payload.get("skills", []),
                    "education": payload.get("education", ""),
                    "companies": payload.get("companies", []),
                    "profile_source": payload.get("document_id", ""),
                }
                break

    if not candidate_profile:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Get interviews
    interview_results = get_candidate_interviews(candidate_id)
    interviews = []
    if interview_results and interview_results[0]:
        for point in interview_results[0]:
            payload = point.payload
            interviews.append(
                {
                    "interview_id": payload.get("interview_id", ""),
                    "interview_date": payload.get("interview_date", ""),
                    "interview_stage": payload.get("interview_stage", ""),
                    "text": payload.get("text", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "section": payload.get("section", ""),
                }
            )

    return {
        "candidate": candidate_profile,
        "interviews": interviews,
        "interview_count": len(interviews),
    }


@router.get("/api/candidates/{candidate_id}/evidence")
async def get_candidate_evidence(candidate_id: str):
    """Get all evidence (profile + interview) for a candidate."""
    # Get profile chunks
    profile_results = get_all_candidates(limit=1000)
    profile_evidence = []

    if profile_results and profile_results[0]:
        for point in profile_results[0]:
            payload = point.payload
            if payload.get("candidate_id") == candidate_id:
                profile_evidence.append(
                    {
                        "text": payload.get("chunk_text", ""),
                        "source": "profile",
                        "source_file": payload.get("document_id", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                    }
                )

    # Get interview chunks
    interview_results = get_candidate_interviews(candidate_id)
    interview_evidence = []

    if interview_results and interview_results[0]:
        for point in interview_results[0]:
            payload = point.payload
            interview_evidence.append(
                {
                    "text": payload.get("text", ""),
                    "source": "interview",
                    "source_file": payload.get("interview_id", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "section": payload.get("section", ""),
                }
            )

    return {
        "candidate_id": candidate_id,
        "profile_evidence": profile_evidence,
        "interview_evidence": interview_evidence,
    }
