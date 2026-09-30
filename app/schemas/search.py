"""Pydantic models for search requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Natural language search query", min_length=1)
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    debug: bool = Field(default=False, description="Include debug information")


class InterpretedRequirements(BaseModel):
    """Structured requirements parsed from natural language query."""

    job_titles: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    experience_min: Optional[float] = None
    experience_max: Optional[float] = None
    location: list[str] = Field(default_factory=list)
    interview_required_skills: list[str] = Field(default_factory=list)
    employment_type: Optional[str] = None
    semantic_query: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A single piece of evidence."""

    text: str
    source: str  # "profile" or "interview"
    source_file: str = ""
    timestamp: str = ""
    section: str = ""
    evidence_type: str = "mentioned"  # mentioned, explained, demonstrated, implemented, uncertain


class CandidateResult(BaseModel):
    """Single candidate search result."""

    rank: int
    candidate_id: str
    name: str
    title: str = ""
    experience: float = 0.0
    location: str = ""
    relevance_score: float
    profile_score: float = 0.0
    interview_score: float = 0.0
    structured_score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    profile_evidence: list[EvidenceItem] = Field(default_factory=list)
    interview_evidence: list[EvidenceItem] = Field(default_factory=list)
    summary: str = ""
    interview_date: Optional[str] = None
    interview_stage: str = ""
    interview_score: Optional[float] = None
    source_documents: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    interpreted_requirements: InterpretedRequirements
    results: list[CandidateResult]
    result_count: int
    debug_info: Optional[dict] = None
