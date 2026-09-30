"""Generic API response models."""

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant: str
    embedding_model: str
    llm: str
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    """Upload response model."""

    success: bool
    message: str
    candidate_id: str = ""
    document_type: str = ""
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: list[str] = []


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str = ""
    suggestion: str = ""


class IngestionStatusResponse(BaseModel):
    """Ingestion status response."""

    total_candidates: int
    total_interviews: int
    total_profile_chunks: int
    total_interview_chunks: int
    status: str
