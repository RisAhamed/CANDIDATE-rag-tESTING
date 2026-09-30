"""Tests for the embedding service."""

import math

import numpy as np
import pytest

from app.embeddings.service import EmbeddingService


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    @pytest.fixture
    def mock_service(self, monkeypatch):
        """Create an EmbeddingService with a mocked SentenceTransformer."""
        # Prevent the singleton from loading a real model
        EmbeddingService._instance = None
        EmbeddingService._model = None

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": lambda self, texts, **kwargs: np.random.randn(
                len(texts) if isinstance(texts, list) else 1, 384
            ).astype(np.float32),
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service = EmbeddingService()
        return service

    def test_singleton_pattern(self, monkeypatch):
        """Test that EmbeddingService is a singleton."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": lambda self, texts, **kwargs: np.zeros((1, 384)),
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service1 = EmbeddingService()
        service2 = EmbeddingService()

        assert service1 is service2

    def test_dimension(self, mock_service):
        """Test that dimension is correctly reported."""
        assert mock_service.dimension == 384

    def test_encode_single_text(self, mock_service):
        """Test encoding a single text string."""
        result = mock_service.encode("Hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_encode_returns_list_of_floats(self, mock_service):
        """Test that encode returns a list of floats."""
        result = mock_service.encode("Test text")

        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    def test_encode_batch(self, mock_service):
        """Test batch encoding."""
        texts = ["Text one", "Text two", "Text three"]
        results = mock_service.encode_batch(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)
            assert len(r) == 384

    def test_encode_batch_empty(self, mock_service):
        """Test batch encoding with empty list."""
        results = mock_service.encode_batch([])
        assert results == []

    def test_encode_query(self, mock_service):
        """Test query encoding."""
        result = mock_service.encode_query("Python developer with FastAPI")

        assert isinstance(result, list)
        assert len(result) == 384

    def test_encode_query_same_as_encode(self, mock_service):
        """Test that encode_query produces same format as encode."""
        query = "Test query"
        result_query = mock_service.encode_query(query)
        result_encode = mock_service.encode(query)

        assert len(result_query) == len(result_encode)
        assert isinstance(result_query, list)
        assert isinstance(result_encode, list)

    def test_embeddings_are_normalized(self, monkeypatch):
        """Test that embeddings are L2-normalized."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

        # Create a mock that returns unnormalized vectors
        def mock_encode(texts, normalize_embeddings=False, **kwargs):
            if isinstance(texts, str):
                return np.ones(384, dtype=np.float32) * 5.0
            return np.ones((len(texts), 384), dtype=np.float32) * 5.0

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": mock_encode,
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service = EmbeddingService()
        result = service.encode("test")

        # Check L2 norm is approximately 1.0 (normalized)
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 0.01

    def test_batch_embeddings_normalized(self, monkeypatch):
        """Test that batch embeddings are also normalized."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

        def mock_encode(texts, normalize_embeddings=False, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 384), dtype=np.float32) * 3.0

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": mock_encode,
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service = EmbeddingService()
        results = service.encode_batch(["a", "b", "c"])

        for r in results:
            norm = math.sqrt(sum(x**2 for x in r))
            assert abs(norm - 1.0) < 0.01

    def test_different_texts_produce_different_embeddings(self, monkeypatch):
        """Test that different texts produce different embeddings."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                # Deterministic but different for different texts
                np.random.seed(hash(texts) % (2**32))
                return np.random.randn(384).astype(np.float32)
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": mock_encode,
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service = EmbeddingService()
        result1 = service.encode("Python developer")
        result2 = service.encode("Java developer")

        # Different texts should produce different embeddings
        assert result1 != result2

    def test_same_text_produces_same_embedding(self, monkeypatch):
        """Test that the same text produces the same embedding."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                np.random.seed(hash(texts) % (2**32))
                return np.random.randn(384).astype(np.float32)
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_model = type("MockModel", (), {
            "get_sentence_embedding_dimension": lambda self: 384,
            "encode": mock_encode,
        })()

        monkeypatch.setattr(
            "app.embeddings.service.SentenceTransformer",
            lambda *a, **kw: mock_model,
        )

        service = EmbeddingService()
        result1 = service.encode("Same text")
        result2 = service.encode("Same text")

        assert result1 == result2


class TestMockEmbeddingService:
    """Tests using the mock embedding service from conftest."""

    def test_mock_service_dimension(self, mock_embedding_service):
        """Test mock service reports correct dimension."""
        assert mock_embedding_service.dimension == 384

    def test_mock_service_encode(self, mock_embedding_service):
        """Test mock service encoding."""
        result = mock_embedding_service.encode("test text")
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_mock_service_call_count(self, mock_embedding_service):
        """Test that call count is tracked."""
        assert mock_embedding_service.call_count == 0
        mock_embedding_service.encode("test")
        assert mock_embedding_service.call_count == 1
        mock_embedding_service.encode("test2")
        assert mock_embedding_service.call_count == 2

    def test_mock_service_batch(self, mock_embedding_service):
        """Test mock service batch encoding."""
        texts = ["a", "b", "c"]
        results = mock_embedding_service.encode_batch(texts)
        assert len(results) == 3
        assert mock_embedding_service.call_count == 1

    def test_mock_service_normalized(self, mock_embedding_service):
        """Test that mock embeddings are normalized."""
        result = mock_embedding_service.encode("test")
        norm = math.sqrt(sum(x**2 for x in result))
        assert abs(norm - 1.0) < 0.01

    def test_mock_service_deterministic(self, mock_embedding_service):
        """Test that mock service is deterministic."""
        result1 = mock_embedding_service.encode("deterministic test")
        result2 = mock_embedding_service.encode("deterministic test")
        assert result1 == result2

    def test_mock_service_different_texts(self, mock_embedding_service):
        """Test that different texts produce different embeddings."""
        result1 = mock_embedding_service.encode("text one")
        result2 = mock_embedding_service.encode("text two")
        assert result1 != result2

    def test_mock_service_reset(self, mock_embedding_service):
        """Test that reset clears tracking."""
        mock_embedding_service.encode("test")
        assert mock_embedding_service.call_count == 1
        mock_embedding_service.reset()
        assert mock_embedding_service.call_count == 0
        assert len(mock_embedding_service.encoded_texts) == 0
