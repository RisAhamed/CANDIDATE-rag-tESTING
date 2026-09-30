"""Bulk ingest candidate documents from a folder into Qdrant."""

import os
import sys
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
from qdrant_client import models

from app.document_parser import parse_document
from app.profile_extractor import extract_profile
from app.qdrant_store import client, create_collection, COLLECTION_NAME


def ingest_folder(folder_path: str):
    """Ingest all supported documents from a folder."""
    create_collection()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    supported_extensions = (".pdf", ".docx", ".txt", ".md")
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ]

    if not files:
        print(f"No supported documents found in {folder_path}")
        return

    print(f"Found {len(files)} document(s) to process")

    success_count = 0
    error_count = 0

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "rb") as f:
                text = parse_document(f, file_name)

            if not text.strip():
                print(f"  [WARN] No text extracted from {file_name}")
                error_count += 1
                continue

            file_type = file_name.split(".")[-1].lower()
            profile = extract_profile(
                text=text,
                file_name=file_name,
                file_type=file_type,
            )

            search_text = profile.to_search_text()
            vector = model.encode(
                search_text,
                normalize_embeddings=True
            ).tolist()

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=profile.to_payload(),
                    )
                ],
            )

            print(f"  [OK] {file_name} -> {profile.name or 'Unknown'}")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {file_name}: {str(e)}")
            error_count += 1

    print(f"\nIngestion complete: {success_count} succeeded, {error_count} failed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_candidates.py <folder_path>")
        print("Example: python scripts/ingest_candidates.py data/candidates/")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    ingest_folder(folder_path)
