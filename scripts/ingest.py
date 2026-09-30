"""Ingest demo data into Qdrant."""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.embeddings.service import embedding_service
from app.ingestion.loaders import load_document
from app.ingestion.parser import normalize_text, parse_candidate_profile
from app.ingestion.chunker import chunk_text, chunk_interview_transcript
from app.qdrant.client import qdrant_client
from app.qdrant.collections import create_collections
from qdrant_client import models


def ingest_profiles():
    """Ingest all candidate profiles from data/candidates/."""
    candidates_dir = os.path.join(settings.BASE_DIR, "data", "candidates")
    if not os.path.isdir(candidates_dir):
        print(f"No candidates directory found: {candidates_dir}")
        return

    files = [f for f in os.listdir(candidates_dir) if f.endswith((".txt", ".pdf", ".docx"))]
    print(f"Found {len(files)} candidate profile(s)")

    for filename in files:
        filepath = os.path.join(candidates_dir, filename)
        # Extract candidate_id from filename (e.g., CAND_001_Arjun_Kumar.txt -> CAND_001)
        candidate_id = filename.split("_")[0] + "_" + filename.split("_")[1]

        try:
            with open(filepath, "rb") as f:
                text, _ = load_document(f.read(), filename)

            text = normalize_text(text)
            profile = parse_candidate_profile(text, filename)
            profile["candidate_id"] = candidate_id

            # Chunk and embed
            chunks = chunk_text(text)
            embeddings = embedding_service.encode_batch([c.text for c in chunks])

            # Store in Qdrant
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

            qdrant_client.client.upsert(
                collection_name=settings.PROFILE_COLLECTION,
                points=points,
            )

            print(f"  [OK] {filename} -> {profile['name']} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")


def ingest_interviews():
    """Ingest all interview transcripts from data/interviews/."""
    interviews_dir = os.path.join(settings.BASE_DIR, "data", "interviews")
    if not os.path.isdir(interviews_dir):
        print(f"No interviews directory found: {interviews_dir}")
        return

    files = [f for f in os.listdir(interviews_dir) if f.endswith((".txt", ".pdf", ".docx", ".srt", ".vtt"))]
    print(f"Found {len(files)} interview transcript(s)")

    for filename in files:
        filepath = os.path.join(interviews_dir, filename)
        # Extract candidate_id from filename
        candidate_id = filename.split("_")[0] + "_" + filename.split("_")[1]
        interview_id = str(uuid.uuid4())

        try:
            with open(filepath, "rb") as f:
                text, _ = load_document(f.read(), filename)

            text = normalize_text(text)

            # Chunk and embed
            chunks = chunk_interview_transcript(text)
            embeddings = embedding_service.encode_batch([c.text for c in chunks])

            # Store in Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "candidate_id": candidate_id,
                            "candidate_name": "",
                            "interview_id": interview_id,
                            "interview_date": "",
                            "source_type": "interview",
                            "chunk_index": i,
                            "text": chunk.text,
                            "timestamp": chunk.timestamp,
                            "interview_stage": "",
                            "section": chunk.section,
                            "metadata": chunk.metadata,
                        },
                    )
                )

            qdrant_client.client.upsert(
                collection_name=settings.INTERVIEW_COLLECTION,
                points=points,
            )

            print(f"  [OK] {filename} -> {candidate_id} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")


if __name__ == "__main__":
    print("Initializing Qdrant collections...")
    create_collections()

    print("\nIngesting candidate profiles...")
    ingest_profiles()

    print("\nIngesting interview transcripts...")
    ingest_interviews()

    print("\nIngestion complete!")
    from app.qdrant.collections import get_collection_stats
    stats = get_collection_stats()
    for name, info in stats.items():
        print(f"  {name}: {info}")
