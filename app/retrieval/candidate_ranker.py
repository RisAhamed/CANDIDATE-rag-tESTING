"""Candidate aggregation, scoring, and ranking.

This module is the core of the candidate search pipeline.  It combines
profile-level and interview-level vector similarity scores with a
structured match score (skills, experience, location) to produce a
single deterministic relevance ranking.
"""

from app.core.config import settings
from app.core.logging import logger
from app.llm.summarizer import generate_candidate_summary
from app.schemas.search import (
    CandidateResult,
    EvidenceItem,
    InterpretedRequirements,
)
from app.retrieval.interview_retriever import retrieve_interviews
from app.retrieval.profile_retriever import retrieve_profiles


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _normalize_score(score: float) -> float:
    """Clamp a similarity score to the [0, 1] range.

    Cosine similarity on L2-normalised embeddings is already in [-1, 1]
    but in practice text embeddings yield values in [0, 1].  Clamping
    guarantees a valid range regardless of the embedding model used.
    """
    return max(0.0, min(1.0, float(score)))


def _calculate_structured_score(
    profile_payload: dict,
    requirements: InterpretedRequirements,
) -> tuple[float, list[str], list[str]]:
    """Compute a structured match score from hard requirements.

    The score is a weighted combination of three sub-scores:

    * **Skills match (50 %)** – fraction of ``required_skills`` that
      appear in the candidate's skill list.
    * **Experience match (30 %)** – ``1.0`` when the candidate's years
      of experience fall inside the ``[experience_min, experience_max]``
      range, ``0.0`` otherwise.
    * **Location match (20 %)** – ``1.0`` when the candidate's location
      matches any of the required locations (substring match,
      case-insensitive).

    When a requirement category is not specified (e.g. no required
    skills) the corresponding sub-score defaults to ``1.0`` so that
    the absence of a filter does not penalise the candidate.

    Args:
        profile_payload: Raw Qdrant payload for a single candidate profile.
        requirements: Structured requirements parsed from the query.

    Returns:
        A tuple ``(structured_score, matched_skills, missing_requirements)``.
    """
    matched_skills: list[str] = []
    missing_requirements: list[str] = []

    # -- Skills (50 %) -------------------------------------------------------
    skills_score = 0.0
    if requirements.required_skills:
        candidate_skills = {s.lower() for s in profile_payload.get("skills", [])}
        for skill in requirements.required_skills:
            if skill.lower() in candidate_skills:
                matched_skills.append(skill)
            else:
                missing_requirements.append(f"Missing skill: {skill}")
        skills_score = len(matched_skills) / len(requirements.required_skills)

    # -- Experience (30 %) ---------------------------------------------------
    experience_score = 0.0
    candidate_experience = float(profile_payload.get("experience", 0.0))
    if requirements.experience_min is not None or requirements.experience_max is not None:
        exp_min = (
            requirements.experience_min
            if requirements.experience_min is not None
            else 0.0
        )
        exp_max = (
            requirements.experience_max
            if requirements.experience_max is not None
            else float("inf")
        )
        if exp_min <= candidate_experience <= exp_max:
            experience_score = 1.0
        else:
            missing_requirements.append(
                f"Experience mismatch: {candidate_experience} years "
                f"(required: {exp_min}-{exp_max})"
            )

    # -- Location (20 %) -----------------------------------------------------
    location_score = 0.0
    if requirements.location:
        candidate_location = profile_payload.get("location", "").lower()
        required_locations = [loc.lower() for loc in requirements.location]
        if any(
            loc in candidate_location or candidate_location in loc
            for loc in required_locations
        ):
            location_score = 1.0
        else:
            missing_requirements.append(
                f"Location mismatch: {profile_payload.get('location', 'Unknown')}"
            )

    # -- Weighted combination ------------------------------------------------
    structured_score = (
        0.5 * skills_score
        + 0.3 * experience_score
        + 0.2 * location_score
    )

    return structured_score, matched_skills, missing_requirements


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_candidates(
    profile_results: list[dict],
    interview_results: list[dict],
    requirements: InterpretedRequirements,
) -> list[CandidateResult]:
    """Aggregate and rank candidates from profile and interview results.

    For every candidate that appears in *either* result set the function
    computes:

    * **profile_score** – the maximum cosine-similarity score across all
      profile hits for that candidate, normalised to ``[0, 1]``.
    * **interview_score** – the maximum cosine-similarity score across
      all interview hits for that candidate, normalised to ``[0, 1]``.
      When no interview evidence exists the score is ``0.0`` (the
      candidate is neither penalised nor boosted).
    * **structured_score** – a deterministic blend of skills (50 %),
      experience (30 %), and location (20 %) match.
    * **final_score** – the weighted sum using ``settings.PROFILE_WEIGHT``,
      ``settings.INTERVIEW_WEIGHT``, and ``settings.STRUCTURED_WEIGHT``.

    Candidates are sorted by ``final_score`` in descending order and the
    top ``settings.FINAL_TOP_K`` are returned as :class:`CandidateResult`
    objects with summaries generated via the LLM summariser.

    Args:
        profile_results: Output of :func:`retrieve_profiles`.
        interview_results: Output of :func:`retrieve_interviews`.
        requirements: Structured requirements parsed from the query.

    Returns:
        A list of up to ``settings.FINAL_TOP_K`` :class:`CandidateResult`
        objects, sorted by relevance (highest first), with ``rank`` fields
        assigned sequentially starting at 1.
    """
    # Group results by candidate_id
    profile_by_candidate: dict[str, list[dict]] = {}
    for r in profile_results:
        profile_by_candidate.setdefault(r["candidate_id"], []).append(r)

    interview_by_candidate: dict[str, list[dict]] = {}
    for r in interview_results:
        interview_by_candidate.setdefault(r["candidate_id"], []).append(r)

    # Union of all candidate IDs
    all_candidate_ids = set(profile_by_candidate) | set(interview_by_candidate)

    results: list[CandidateResult] = []

    for cid in all_candidate_ids:
        # -- Profile score ---------------------------------------------------
        profiles = profile_by_candidate.get(cid, [])
        if profiles:
            best_profile = max(profiles, key=lambda x: x["score"])
            profile_score = _normalize_score(best_profile["score"])
            profile_payload: dict = best_profile["payload"]
        else:
            profile_score = 0.0
            profile_payload = {}

        # -- Interview score --------------------------------------------------
        interviews = interview_by_candidate.get(cid, [])
        if interviews:
            best_interview = max(interviews, key=lambda x: x["score"])
            interview_score = _normalize_score(best_interview["score"])
        else:
            interview_score = 0.0
            best_interview = None

        # -- Structured score -------------------------------------------------
        structured_score, matched_skills, missing_requirements = (
            _calculate_structured_score(profile_payload, requirements)
        )

        # -- Final weighted score ---------------------------------------------
        final_score = (
            settings.PROFILE_WEIGHT * profile_score
            + settings.INTERVIEW_WEIGHT * interview_score
            + settings.STRUCTURED_WEIGHT * structured_score
        )

        # -- Evidence ---------------------------------------------------------
        profile_evidence: list[EvidenceItem] = []
        for p in profiles:
            payload = p["payload"]
            profile_evidence.append(
                EvidenceItem(
                    text=payload.get("chunk_text", payload.get("text", f"Profile match (score: {p['score']:.3f})")),
                    source="profile",
                    source_file=payload.get("document_id", payload.get("source_file", "")),
                    timestamp=payload.get("timestamp", ""),
                    section=payload.get("section", "profile"),
                    evidence_type="mentioned",
                )
            )

        interview_evidence: list[EvidenceItem] = []
        for iv in interviews:
            payload = iv["payload"]
            interview_evidence.append(
                EvidenceItem(
                    text=payload.get("text", payload.get("content", "")),
                    source="interview",
                    source_file=payload.get("source_file", ""),
                    timestamp=payload.get("timestamp", ""),
                    section=payload.get("section", ""),
                    evidence_type=payload.get("evidence_type", "mentioned"),
                )
            )

        # -- Assemble CandidateResult -----------------------------------------
        candidate_result = CandidateResult(
            rank=0,  # assigned after sorting
            candidate_id=cid,
            name=profile_payload.get("candidate_name", profile_payload.get("name", "")),
            title=profile_payload.get("title", ""),
            experience=float(profile_payload.get("experience", 0.0)),
            location=profile_payload.get("location", ""),
            relevance_score=final_score,
            profile_score=profile_score,
            interview_score=interview_score,
            structured_score=structured_score,
            matched_skills=matched_skills,
            missing_requirements=missing_requirements,
            profile_evidence=profile_evidence,
            interview_evidence=interview_evidence,
            interview_date=(
                best_interview["payload"].get("interview_date")
                if best_interview
                else None
            ),
            interview_stage=(
                best_interview["payload"].get("stage", "")
                if best_interview
                else ""
            ),
            source_documents=list(
                {
                    e.source_file
                    for e in profile_evidence
                    if e.source_file
                }
                | {
                    e.source_file
                    for e in interview_evidence
                    if e.source_file
                }
            ),
        )

        results.append(candidate_result)

    # Sort by final_score descending (deterministic)
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    # Truncate to FINAL_TOP_K and assign ranks
    final_results = results[: settings.FINAL_TOP_K]
    for i, r in enumerate(final_results, start=1):
        r.rank = i

    # Generate LLM summaries for the top candidates
    for r in final_results:
        try:
            candidate_data = {
                "name": r.name,
                "title": r.title,
                "experience_years": r.experience,
                "location": r.location,
                "skills": r.matched_skills,
                "profile_evidence": [e.text for e in r.profile_evidence],
                "interview_evidence": [e.text for e in r.interview_evidence],
            }
            r.summary = generate_candidate_summary(candidate_data, requirements)
        except Exception as exc:
            logger.warning(
                "Failed to generate summary for candidate %s: %s",
                r.candidate_id,
                exc,
            )
            r.summary = ""

    if final_results:
        logger.info(
            "Ranked %d candidates, returning top %d. Top score: %.3f",
            len(results),
            len(final_results),
            final_results[0].relevance_score,
        )
    else:
        logger.info("No candidates found for ranking")

    return final_results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def search_pipeline(
    query: str,
    requirements: InterpretedRequirements,
) -> list[CandidateResult]:
    """Orchestrate the end-to-end search pipeline.

    1. Retrieve candidate profiles via semantic + structured search.
    2. Retrieve interview chunks scoped to the profile candidates
       (or across all interviews when no profiles are found).
    3. Aggregate, score, and rank the candidates.

    Args:
        query: Natural language search query.
        requirements: Structured requirements parsed from the query.

    Returns:
        A list of up to ``settings.FINAL_TOP_K`` :class:`CandidateResult`
        objects sorted by relevance.
    """
    logger.info("Starting search pipeline for query: %s...", query[:100])

    # Step 1 – profile retrieval
    profile_results = retrieve_profiles(query, requirements)

    # Step 2 – interview retrieval (scoped to profile candidates when possible)
    candidate_ids = [r["candidate_id"] for r in profile_results]
    if candidate_ids:
        interview_results = retrieve_interviews(
            query, requirements, candidate_ids=candidate_ids
        )
    else:
        interview_results = retrieve_interviews(query, requirements)

    # Step 3 – ranking
    ranked = rank_candidates(profile_results, interview_results, requirements)

    logger.info("Search pipeline complete. Found %d candidates.", len(ranked))
    return ranked
