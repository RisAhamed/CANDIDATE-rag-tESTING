"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Application settings loaded from environment variables."""

    # Base directory
    BASE_DIR: Path = BASE_DIR

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Collections
    PROFILE_COLLECTION: str = os.getenv("PROFILE_COLLECTION", "candidate_profiles")
    INTERVIEW_COLLECTION: str = os.getenv("INTERVIEW_COLLECTION", "interview_chunks")

    # Ranking weights
    PROFILE_WEIGHT: float = float(os.getenv("PROFILE_WEIGHT", "0.45"))
    INTERVIEW_WEIGHT: float = float(os.getenv("INTERVIEW_WEIGHT", "0.35"))
    STRUCTURED_WEIGHT: float = float(os.getenv("STRUCTURED_WEIGHT", "0.20"))

    # Retrieval
    TOP_K_PROFILE: int = int(os.getenv("TOP_K_PROFILE", "20"))
    TOP_K_INTERVIEW: int = int(os.getenv("TOP_K_INTERVIEW", "50"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "10"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Upload
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # SQLite
    SQLITE_PATH: Path = BASE_DIR / "data" / "candidates.db"

    # Allowed extensions
    PROFILE_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}
    INTERVIEW_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx", ".md", ".srt", ".vtt"}

    # Debug
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for the configured model."""
        # bge-small-en-v1.5 has 384 dimensions
        if "bge-small" in self.EMBEDDING_MODEL:
            return 384
        elif "bge-base" in self.EMBEDDING_MODEL:
            return 768
        elif "bge-large" in self.EMBEDDING_MODEL:
            return 1024
        elif "MiniLM" in self.EMBEDDING_MODEL:
            return 384
        return 384  # default

    def validate(self) -> list[str]:
        """Validate settings and return list of issues."""
        issues = []
        if not self.GROQ_API_KEY:
            issues.append("GROQ_API_KEY is not set")
        if self.PROFILE_WEIGHT + self.INTERVIEW_WEIGHT + self.STRUCTURED_WEIGHT != 1.0:
            issues.append("Ranking weights must sum to 1.0")
        return issues


settings = Settings()
