import json
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "jobs"

client = QdrantClient(url=QDRANT_URL)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def build_search_text(job):
    skills = ", ".join(job["skills"])

    return f"""
    Job title: {job["title"]}
    Company: {job["company"]}
    Location: {job["location"]}
    Skills: {skills}
    Description: {job["description"]}
    """.strip()


def ingest():
    with open("data/jobs.jsonl", "r", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]

    points = []

    for job in jobs:

        search_text = build_search_text(job)

        vector = model.encode(
            search_text,
            normalize_embeddings=True
        ).tolist()

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    **job,
                    "search_text": search_text
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Inserted {len(points)} jobs")


if __name__ == "__main__":
    ingest()