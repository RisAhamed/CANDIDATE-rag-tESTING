"""Text chunking with configurable size and overlap."""

import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import logger


# Known skills list for extraction
KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    "bash", "shell", "powershell", "sql", "html", "css", "sass", "less",
    "fastapi", "flask", "django", "react", "angular", "vue", "vue.js", "svelte",
    "next.js", "nuxt", "express", "express.js", "spring", "spring boot",
    "rails", "laravel", "symfony", "asp.net", ".net", "dotnet",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "scipy", "matplotlib", "seaborn", "plotly", "opencv", "nltk",
    "spacy", "transformers", "hugging face", "huggingface", "langchain",
    "llamaindex", "streamlit", "dash", "gradio",
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "mongo",
    "redis", "elasticsearch", "cassandra", "dynamodb", "firebase",
    "supabase", "neo4j", "couchdb", "mariadb", "oracle", "sql server",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "travis ci", "puppet", "chef", "vagrant", "helm",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data engineering", "data analysis",
    "etl", "airflow", "spark", "hadoop", "kafka", "rabbitmq", "dbt",
    "snowflake", "databricks", "bigquery", "redshift", "looker", "tableau",
    "power bi", "mlflow", "kubeflow", "ray", "mlops", "llm", "rag",
    "retrieval augmented generation", "embeddings", "vector database",
    "qdrant", "pinecone", "weaviate", "chroma", "milvus", "faiss",
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "agile",
    "scrum", "kanban", "ci/cd", "rest api", "restful", "graphql", "grpc",
    "websocket", "microservices", "serverless", "linux", "unix", "nginx",
    "apache", "tomcat", "celery", "memcached", "oauth", "jwt", "sso",
    "ldap", "testing", "unit testing", "integration testing", "selenium",
    "pytest", "jest", "cypress", "tdd", "bdd", "code review",
    "pair programming", "mentoring", "leadership", "communication",
    "project management", "product management", "stakeholder management",
    "client management", "team management", "recruiting", "interviewing",
    "onboarding", "training",
}


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    section: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text.strip():
        return []

    # Split into sentences first
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_index = 0
    char_position = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence exceeds chunk size, save current chunk
        if current_size + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                )
            )
            chunk_index += 1
            char_position += len(chunk_text) + 1

            # Keep overlap sentences
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_size += len(s)

            current_chunk = overlap_sentences
            current_size = overlap_size

        current_chunk.append(sentence)
        current_size += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            Chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=char_position,
                end_char=char_position + len(chunk_text),
            )
        )

    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks


def chunk_interview_transcript(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Chunk interview transcript preserving speaker labels and structure.

    Keeps interviewer/candidate exchanges together when possible.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text.strip():
        return []

    # Split by speaker turns
    speaker_blocks = split_by_speaker(text)
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_index = 0

    for block in speaker_blocks:
        block_len = len(block["text"])

        if current_size + block_len > chunk_size and current_chunk:
            chunk_text = "\n".join(b["text"] for b in current_chunk)
            # Get timestamp from first block
            timestamp = current_chunk[0].get("timestamp", "")
            section = current_chunk[0].get("speaker", "")

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_text),
                    section=section,
                    timestamp=timestamp,
                    metadata={"speakers": list(set(b.get("speaker", "") for b in current_chunk))},
                )
            )
            chunk_index += 1

            # Keep overlap
            overlap_blocks = []
            overlap_size = 0
            for b in reversed(current_chunk):
                if overlap_size + len(b["text"]) > overlap:
                    break
                overlap_blocks.insert(0, b)
                overlap_size += len(b["text"])

            current_chunk = overlap_blocks
            current_size = overlap_size

        current_chunk.append(block)
        current_size += block_len

    if current_chunk:
        chunk_text = "\n".join(b["text"] for b in current_chunk)
        timestamp = current_chunk[0].get("timestamp", "")
        section = current_chunk[0].get("speaker", "")

        chunks.append(
            Chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                section=section,
                timestamp=timestamp,
                metadata={"speakers": list(set(b.get("speaker", "") for b in current_chunk))},
            )
        )

    logger.info(f"Created {len(chunks)} interview chunks")
    return chunks


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_speaker(text: str) -> list[dict]:
    """
    Split interview transcript by speaker turns.

    Returns list of dicts with 'speaker', 'text', 'timestamp' keys.
    """
    blocks = []
    lines = text.split("\n")
    current_speaker = ""
    current_text = []
    current_timestamp = ""

    speaker_pattern = re.compile(
        r"^(INTERVIEWER|CANDIDATE|INTERVIEWEE|SPEAKER\s*\d*|Q|A)\s*:\s*",
        re.IGNORECASE,
    )
    timestamp_pattern = re.compile(r"\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for timestamp
        ts_match = timestamp_pattern.search(line)
        if ts_match:
            current_timestamp = ts_match.group(1)

        # Check for speaker label
        speaker_match = speaker_pattern.match(line)
        if speaker_match:
            # Save previous block
            if current_text and current_speaker:
                blocks.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "timestamp": current_timestamp,
                })
            current_speaker = speaker_match.group(1).upper()
            current_text = [line[speaker_match.end():].strip()]
            current_timestamp = current_timestamp or ts_match.group(1) if ts_match else ""
        else:
            current_text.append(line)

    # Save last block
    if current_text and current_speaker:
        blocks.append({
            "speaker": current_speaker,
            "text": " ".join(current_text),
            "timestamp": current_timestamp,
        })

    # If no speaker labels found, treat as single block
    if not blocks:
        blocks.append({
            "speaker": "UNKNOWN",
            "text": text,
            "timestamp": "",
        })

    return blocks
