"""Pydantic models for interview data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class InterviewBase(BaseModel):
    """Base interview model."""

    interview_id: str = Field(..., description="Unique interview identifier")
    candidate_id: str = Field(..., description="Associated candidate ID")
    interview_date: Optional[str] = Field(default=None, description="Interview date")
    interviewer: str = Field(default="", description="Interviewer name")
    interview_stage: str = Field(default="", description="Interview stage (e.g., technical, HR)")
    transcript_text: str = Field(default="", description="Full transcript text")
    score: Optional[float] = Field(default=None, description="Interview score")
    recommendation: str = Field(default="", description="Interviewer recommendation")
    source_file: str = Field(default="", description="Source file")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InterviewCreate(BaseModel):
    """Model for creating an interview record."""

    interview_id: str
    candidate_id: str
    interview_date: Optional[str] = None
    interviewer: str = ""
    interview_stage: str = ""
    transcript_text: str = ""
    score: Optional[float] = None
    recommendation: str = ""
    source_file: str = ""


class InterviewChunk(BaseModel):
    """Model for interview chunks."""

    chunk_id: str
    candidate_id: str
    interview_id: str
    chunk_text: str
    section: str = ""
    timestamp: str = ""
    metadata: dict = Field(default_factory=dict)


class InterviewResponse(BaseModel):
    """Interview response model."""

    interview_id: str
    candidate_id: str
    candidate_name: str = ""
    interview_date: Optional[str] = None
    interviewer: str = ""
    interview_stage: str = ""
    score: Optional[float] = None
    recommendation: str = ""
    source_file: str = ""
    chunk_count: int = 0
    created_at: Optional[datetime] = None
