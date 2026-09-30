"""Pydantic models for candidate data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CandidateBase(BaseModel):
    """Base candidate model."""

    candidate_id: str = Field(..., description="Unique candidate identifier")
    name: str = Field(default="", description="Candidate full name")
    email: str = Field(default="", description="Candidate email")
    phone: str = Field(default="", description="Candidate phone")
    location: str = Field(default="", description="Candidate location")
    current_title: str = Field(default="", description="Current job title")
    total_experience: float = Field(default=0.0, description="Total years of experience")
    skills: list[str] = Field(default_factory=list, description="Candidate skills")
    education: str = Field(default="", description="Education information")
    companies: list[str] = Field(default_factory=list, description="Previous companies")
    resume_text: str = Field(default="", description="Full resume text")
    profile_source: str = Field(default="", description="Source file for profile")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateCreate(BaseModel):
    """Model for creating a candidate."""

    candidate_id: str
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    current_title: str = ""
    total_experience: float = 0.0
    skills: list[str] = []
    education: str = ""
    companies: list[str] = []
    resume_text: str = ""
    profile_source: str = ""


class CandidateResponse(BaseModel):
    """Candidate response model."""

    candidate_id: str
    name: str
    email: str
    phone: str
    location: str
    current_title: str
    total_experience: float
    skills: list[str]
    education: str
    companies: list[str]
    profile_source: str
    has_interviews: bool = False
    interview_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CandidateDetailResponse(BaseModel):
    """Detailed candidate response with interviews."""

    candidate: CandidateResponse
    interviews: list[dict] = []
    interview_evidence: list[dict] = []
