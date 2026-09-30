"""Upload API endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.service import embedding_service
from app.ingestion.pipeline import ingest_interview, ingest_profile
from app.qdrant.client import qdrant_client
from app.schemas.responses import UploadResponse

router = APIRouter()


def validate_file(filename: str, allowed_extensions: set[str]) -> str:
    """Validate file extension and return it."""
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {allowed_extensions}",
        )
    return ext


def save_upload(file: UploadFile, file_bytes: bytes) -> Path:
    """Save uploaded file to disk."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.UPLOAD_DIR / file.filename

    # Prevent path traversal
    if not str(file_path.resolve()).startswith(str(settings.UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path


@router.post("/api/candidates/upload", response_model=UploadResponse)
async def upload_candidate_profile(
    file: UploadFile = File(...),
    candidate_id: str = Form(...),
):
    """
    Upload and process a candidate profile/resume.

    Extracts text, parses profile, generates embeddings, stores in Qdrant.
    """
    logger.info(f"Profile upload: {candidate_id}, file: {file.filename}")

    # Validate
    validate_file(file.filename, settings.PROFILE_EXTENSIONS)

    # Read file
    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Save file
        save_upload(file, file_bytes)

        # Ingest profile
        result = ingest_profile(
            file_bytes=file_bytes,
            filename=file.filename,
            candidate_id=candidate_id,
            embedding_service=embedding_service,
            qdrant_client=qdrant_client.client,
        )

        return UploadResponse(
            success=True,
            message=f"Profile processed for {result['name'] or candidate_id}",
            candidate_id=candidate_id,
            document_type="profile",
            chunks_created=result["chunks_created"],
            embeddings_generated=result["embeddings_generated"],
        )

    except Exception as e:
        logger.error(f"Profile upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/api/interviews/upload", response_model=UploadResponse)
async def upload_interview(
    file: UploadFile = File(...),
    candidate_id: str = Form(...),
    interview_id: str = Form(default=""),
    interview_date: str = Form(default=""),
    interviewer: str = Form(default=""),
    interview_stage: str = Form(default=""),
):
    """
    Upload and process an interview transcript/report.

    Extracts text, chunks transcript, generates embeddings, stores in Qdrant.
    """
    logger.info(f"Interview upload: {candidate_id}, file: {file.filename}")

    # Validate
    validate_file(file.filename, settings.INTERVIEW_EXTENSIONS)

    # Read file
    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Save file
        save_upload(file, file_bytes)

        # Ingest interview
        result = ingest_interview(
            file_bytes=file_bytes,
            filename=file.filename,
            candidate_id=candidate_id,
            embedding_service=embedding_service,
            qdrant_client=qdrant_client.client,
            interview_id=interview_id or None,
            interview_date=interview_date,
            interviewer=interviewer,
            interview_stage=interview_stage,
        )

        return UploadResponse(
            success=True,
            message=f"Interview processed for {candidate_id}",
            candidate_id=candidate_id,
            document_type="interview",
            chunks_created=result["chunks_created"],
            embeddings_generated=result["embeddings_generated"],
        )

    except Exception as e:
        logger.error(f"Interview upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
