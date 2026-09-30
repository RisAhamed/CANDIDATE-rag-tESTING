"""Pytest configuration and shared fixtures for Candidate Search RAG tests."""

import numpy as np
import pytest

from app.schemas.search import InterpretedRequirements


# ---------------------------------------------------------------------------
# Mock Embedding Service
# ---------------------------------------------------------------------------


class MockEmbeddingService:
    """Deterministic mock embedding service for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._call_count = 0
        self._encoded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def encoded_texts(self) -> list[str]:
        return list(self._encoded_texts)

    def _text_to_vector(self, text: str) -> list[float]:
        """Create a deterministic pseudo-embedding from text."""
        # Use a simple hash-based approach for deterministic vectors
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(self._dimension).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def encode(self, text: str) -> list[float]:
        """Encode a single text."""
        self._call_count += 1
        self._encoded_texts.append(text)
        return self._text_to_vector(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts."""
        self._call_count += 1
        self._encoded_texts.extend(texts)
        return [self._text_to_vector(t) for t in texts]

    def encode_query(self, query: str) -> list[float]:
        """Encode a search query."""
        return self.encode(query)

    def reset(self):
        """Reset call tracking."""
        self._call_count = 0
        self._encoded_texts.clear()


# ---------------------------------------------------------------------------
# Mock Qdrant Client
# ---------------------------------------------------------------------------


class MockQdrantClient:
    """Mock Qdrant client that returns canned responses."""

    def __init__(self):
        self._profile_results: list = []
        self._interview_results: list = []
        self._upsert_calls: list = []
        self._query_calls: list = []

    def set_profile_results(self, results: list):
        """Set canned profile search results."""
        self._profile_results = results

    def set_interview_results(self, results: list):
        """Set canned interview search results."""
        self._interview_results = results

    def query_points(self, **kwargs):
        """Mock query_points - returns canned results based on collection."""
        self._query_calls.append(kwargs)
        collection = kwargs.get("collection_name", "")

        class MockResponse:
            def __init__(self, points):
                self.points = points

        if "profile" in collection.lower():
            return MockResponse(self._profile_results)
        else:
            return MockResponse(self._interview_results)

    def upsert(self, **kwargs):
        """Mock upsert - records calls."""
        self._upsert_calls.append(kwargs)
        return {"status": "ok"}

    def scroll(self, **kwargs):
        """Mock scroll - returns empty by default."""
        return []

    @property
    def upsert_calls(self) -> list:
        return self._upsert_calls

    @property
    def query_calls(self) -> list:
        return self._query_calls


# ---------------------------------------------------------------------------
# Mock Groq Client
# ---------------------------------------------------------------------------


class MockGroqClient:
    """Mock Groq LLM client."""

    def __init__(self, response: str = "Mock LLM response"):
        self._response = response
        self._calls: list = []

    def chat_completion(self, **kwargs) -> str:
        """Mock chat completion."""
        self._calls.append(kwargs)
        return self._response

    @property
    def calls(self) -> list:
        return self._calls

    def set_response(self, response: str):
        self._response = response


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_service():
    """Provide a fresh mock embedding service."""
    return MockEmbeddingService(dimension=384)


@pytest.fixture
def mock_qdrant_client():
    """Provide a fresh mock Qdrant client."""
    return MockQdrantClient()


@pytest.fixture
def mock_groq_client():
    """Provide a fresh mock Groq client."""
    return MockGroqClient()


@pytest.fixture
def sample_requirements():
    """Provide sample interpreted requirements."""
    return InterpretedRequirements(
        job_titles=["Python Developer"],
        required_skills=["python", "fastapi", "rag"],
        experience_min=3.0,
        experience_max=8.0,
        location=["Bangalore"],
        interview_required_skills=["fastapi", "rag"],
        semantic_query="Python developer with FastAPI and RAG experience",
    )


@pytest.fixture
def sample_requirements_no_constraints():
    """Provide requirements with no hard constraints."""
    return InterpretedRequirements(
        job_titles=[],
        required_skills=[],
        experience_min=None,
        experience_max=None,
        location=[],
        interview_required_skills=[],
        semantic_query="software engineer",
    )


@pytest.fixture
def sample_profile_payload():
    """Provide a sample candidate profile payload (as stored in Qdrant)."""
    return {
        "candidate_id": "CAND_001",
        "candidate_name": "Arjun Kumar",
        "title": "Senior Python Developer",
        "location": "Bangalore",
        "experience": 5.0,
        "skills": ["python", "fastapi", "rag", "langchain", "docker"],
        "source_type": "profile",
        "source_file": "CAND_001_Arjun_Kumar.txt",
        "chunk_index": 0,
        "chunk_text": "Arjun Kumar is a Senior Python Developer with 5 years of experience",
        "section": "summary",
    }


@pytest.fixture
def sample_interview_payload():
    """Provide a sample interview chunk payload (as stored in Qdrant)."""
    return {
        "candidate_id": "CAND_001",
        "candidate_name": "Arjun Kumar",
        "interview_id": "int_001",
        "interview_date": "2024-03-15",
        "source_type": "interview",
        "chunk_index": 0,
        "text": "CANDIDATE: I built a RAG pipeline using FastAPI and vector databases",
        "timestamp": "00:05:30",
        "interview_stage": "technical",
        "section": "INTERVIEWER",
        "metadata": {"speakers": ["INTERVIEWER", "CANDIDATE"]},
    }


@pytest.fixture
def sample_profile_results():
    """Provide sample profile search results."""
    return [
        {
            "candidate_id": "CAND_001",
            "score": 0.92,
            "payload": {
                "candidate_id": "CAND_001",
                "candidate_name": "Arjun Kumar",
                "title": "Senior Python Developer",
                "location": "Bangalore",
                "experience": 5.0,
                "skills": ["python", "fastapi", "rag", "langchain", "docker"],
                "source_file": "CAND_001_Arjun_Kumar.txt",
            },
        },
        {
            "candidate_id": "CAND_002",
            "score": 0.78,
            "payload": {
                "candidate_id": "CAND_002",
                "candidate_name": "Priya Sharma",
                "title": "Python Developer",
                "location": "Mumbai",
                "experience": 5.0,
                "skills": ["python", "django", "postgresql"],
                "source_file": "CAND_002_Priya_Sharma.txt",
            },
        },
        {
            "candidate_id": "CAND_005",
            "score": 0.85,
            "payload": {
                "candidate_id": "CAND_005",
                "candidate_name": "Amit Singh",
                "title": "ML Engineer",
                "location": "Hyderabad",
                "experience": 5.0,
                "skills": ["python", "rag", "langchain", "llm"],
                "source_file": "CAND_005_Amit_Singh.txt",
            },
        },
    ]


@pytest.fixture
def sample_interview_results():
    """Provide sample interview search results."""
    return [
        {
            "candidate_id": "CAND_001",
            "score": 0.88,
            "payload": {
                "candidate_id": "CAND_001",
                "candidate_name": "Arjun Kumar",
                "text": "CANDIDATE: I built a RAG pipeline using FastAPI and vector databases",
                "timestamp": "00:05:30",
                "interview_date": "2024-03-15",
                "stage": "technical",
                "section": "INTERVIEWER",
                "source_file": "CAND_001_interview.txt",
                "evidence_type": "demonstrated",
            },
        },
        {
            "candidate_id": "CAND_005",
            "score": 0.82,
            "payload": {
                "candidate_id": "CAND_005",
                "candidate_name": "Amit Singh",
                "text": "CANDIDATE: I implemented LangChain agents for RAG applications",
                "timestamp": "00:12:00",
                "interview_date": "2024-04-10",
                "stage": "technical",
                "section": "CANDIDATE",
                "source_file": "CAND_005_interview.txt",
                "evidence_type": "implemented",
            },
        },
    ]


@pytest.fixture
def sample_resume_text():
    """Provide a sample resume text for testing."""
    return """Arjun Kumar
Email: arjun.kumar@email.com
Phone: +91-9876543210
Location: Bangalore

Summary
Senior Python Developer with 5 years of experience building scalable web applications and RAG systems.

Skills
Python, FastAPI, Django, RAG, LangChain, Docker, PostgreSQL, Redis, AWS, Git

Experience
Senior Python Developer at TechCorp (2021-Present)
- Built RAG pipeline using FastAPI and vector databases
- Designed microservices architecture serving 1M+ requests/day

Python Developer at StartupXYZ (2019-2021)
- Developed REST APIs with FastAPI and Django
- Implemented caching layer with Redis

Education
B.Tech in Computer Science, IIT Delhi (2015-2019)
"""


@pytest.fixture
def sample_interview_text():
    """Provide a sample interview transcript for testing."""
    return """[00:00:00]
INTERVIEWER: Welcome Arjun, let's start with your experience with FastAPI.

[00:00:15]
CANDIDATE: Thank you. I've been using FastAPI for about 3 years now. At TechCorp, I built a complete RAG pipeline using FastAPI as the API layer.

[00:01:30]
INTERVIEWER: Can you explain how you designed the RAG system?

[00:01:45]
CANDIDATE: Sure. The system uses embeddings to index documents into a vector database. When a query comes in, we retrieve relevant chunks and pass them to an LLM for generation.

[00:03:00]
INTERVIEWER: What embedding model did you use?

[00:03:15]
CANDIDATE: We used BAAI/bge-small-en-v1.5 for its balance of speed and accuracy. The 384-dimensional embeddings worked well for our use case.

[00:05:30]
INTERVIEWER: How did you handle scaling?

[00:05:45]
CANDIDATE: We deployed on Kubernetes with horizontal pod autoscaling. The FastAPI services were stateless, so scaling was straightforward.
"""
