"""Ingestion pipeline: document -> chunks -> embeddings -> Qdrant."""

import uuid
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.chunker import chunk_interview_transcript, chunk_text
from app.ingestion.loaders import load_document
from app.ingestion.parser import normalize_text, parse_candidate_profile


def ingest_profile(
    file_bytes: bytes,
    filename: str,
    candidate_id: str,
    embedding_service,
    qdrant_client,
) -> dict:
    """
    Ingest a candidate profile document.

    Returns dict with ingestion results.
    """
    logger.info(f"Ingesting profile for {candidate_id}: {filename}")

    # Load and parse document
    text, file_type = load_document(file_bytes, filename)
    text = normalize_text(text)

    if not text.strip():
        raise ValueError(f"No text extracted from {filename}")

    # Parse candidate profile
    profile = parse_candidate_profile(text, filename)

    # Update with provided candidate_id
    profile["candidate_id"] = candidate_id

    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings for chunks
    embeddings = []
    for chunk in chunks:
        embedding = embedding_service.encode(chunk.text)
        embeddings.append(embedding)

    # Store in Qdrant
    from qdrant_client import models

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "candidate_id": candidate_id,
                    "candidate_name": profile["name"],
                    "title": profile.get("current_title", ""),
                    "location": profile.get("location", ""),
                    "experience": profile["total_experience"],
                    "skills": profile["skills"],
                    "source_type": "profile",
                    "document_id": filename,
                    "chunk_index": i,
                    "chunk_text": chunk.text,
                    "section": chunk.section,
                },
            )
        )

    qdrant_client.upsert(
        collection_name=settings.PROFILE_COLLECTION,
        points=points,
    )

    logger.info(
        f"Profile ingestion complete: {candidate_id}, "
        f"{len(chunks)} chunks, {len(embeddings)} embeddings"
    )

    return {
        "candidate_id": candidate_id,
        "name": profile["name"],
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
        "skills": profile["skills"],
        "experience": profile["total_experience"],
    }


def ingest_interview(
    file_bytes: bytes,
    filename: str,
    candidate_id: str,
    embedding_service,
    qdrant_client,
    interview_id: str | None = None,
    interview_date: str = "",
    interviewer: str = "",
    interview_stage: str = "",
) -> dict:
    """
    Ingest an interview transcript/document.

    Returns dict with ingestion results.
    """
    interview_id = interview_id or str(uuid.uuid4())
    logger.info(f"Ingesting interview for {candidate_id}: {filename}")

    # Load and parse document
    text, file_type = load_document(file_bytes, filename)
    text = normalize_text(text)

    if not text.strip():
        raise ValueError(f"No text extracted from {filename}")

    # Chunk the transcript
    chunks = chunk_interview_transcript(text)

    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        embedding = embedding_service.encode(chunk.text)
        embeddings.append(embedding)

    # Store in Qdrant
    from qdrant_client import models

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "candidate_id": candidate_id,
                    "candidate_name": "",  # Will be filled from profile
                    "interview_id": interview_id,
                    "interview_date": interview_date,
                    "source_type": "interview",
                    "chunk_index": i,
                    "text": chunk.text,
                    "timestamp": chunk.timestamp,
                    "interview_stage": interview_stage,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                },
            )
        )

    qdrant_client.upsert(
        collection_name=settings.INTERVIEW_COLLECTION,
        points=points,
    )

    logger.info(
        f"Interview ingestion complete: {interview_id}, "
        f"{len(chunks)} chunks, {len(embeddings)} embeddings"
    )

    return {
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
    }
