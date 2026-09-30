"""Tests for search and ranking pipeline."""

import pytest

from app.retrieval.candidate_ranker import rank_candidates, search_pipeline
from app.retrieval.interview_retriever import retrieve_interviews
from app.retrieval.profile_retriever import retrieve_profiles
from app.schemas.search import InterpretedRequirements


# ===========================================================================
# Profile Retrieval Tests
# ===========================================================================


class TestRetrieveProfiles:
    """Tests for profile retrieval."""

    def test_retrieve_profiles_basic(
        self, monkeypatch, mock_embedding_service, mock_qdrant_client, sample_requirements
    ):
        """Test basic profile retrieval."""
        # Mock the search_profiles function
        class MockResult:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        mock_results = [
            MockResult(
                {
                    "candidate_id": "CAND_001",
                    "candidate_name": "Arjun Kumar",
                    "skills": ["python", "fastapi"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
                0.92,
            )
        ]
        mock_qdrant_client.set_profile_results(mock_results)

        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: mock_results,
        )

        results = retrieve_profiles("Python developer", sample_requirements)

        assert len(results) == 1
        assert results[0]["candidate_id"] == "CAND_001"
        assert results[0]["score"] == 0.92
        assert "payload" in results[0]

    def test_retrieve_profiles_with_filters(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test that structured filters are passed correctly."""
        captured_filters = {}

        def mock_search(**kwargs):
            captured_filters.update(kwargs.get("filters", {}))
            return []

        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            mock_search,
        )

        retrieve_profiles("Python", sample_requirements)

        assert captured_filters.get("experience_min") == 3.0
        assert captured_filters.get("experience_max") == 8.0
        assert "python" in captured_filters.get("skills", [])

    def test_retrieve_profiles_no_requirements(
        self, monkeypatch, mock_embedding_service, sample_requirements_no_constraints
    ):
        """Test retrieval with no hard constraints."""
        captured_filters = {}

        def mock_search(**kwargs):
            captured_filters.update(kwargs.get("filters", {}))
            return []

        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            mock_search,
        )

        retrieve_profiles("engineer", sample_requirements_no_constraints)

        # No filters should be set when no requirements
        assert captured_filters == {}

    def test_retrieve_profiles_empty_results(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test handling of empty search results."""
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: [],
        )

        results = retrieve_profiles("Python", sample_requirements)
        assert results == []


# ===========================================================================
# Interview Retrieval Tests
# ===========================================================================


class TestRetrieveInterviews:
    """Tests for interview retrieval."""

    def test_retrieve_interviews_basic(
        self, monkeypatch, mock_embedding_service, mock_qdrant_client, sample_requirements
    ):
        """Test basic interview retrieval."""
        class MockResult:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        mock_results = [
            MockResult(
                {
                    "candidate_id": "CAND_001",
                    "text": "I built a RAG pipeline",
                    "timestamp": "00:05:30",
                },
                0.88,
            )
        ]
        mock_qdrant_client.set_interview_results(mock_results)

        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: mock_results,
        )

        results = retrieve_interviews("RAG pipeline", sample_requirements)

        assert len(results) == 1
        assert results[0]["candidate_id"] == "CAND_001"
        assert results[0]["score"] == 0.88

    def test_retrieve_interviews_scoped_to_candidates(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test that interview retrieval is scoped to specific candidates."""
        captured_candidate_ids = []

        def mock_search(**kwargs):
            captured_candidate_ids.append(kwargs.get("candidate_id"))
            return []

        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            mock_search,
        )

        retrieve_interviews(
            "Python", sample_requirements, candidate_ids=["CAND_001", "CAND_002"]
        )

        assert "CAND_001" in captured_candidate_ids
        assert "CAND_002" in captured_candidate_ids

    def test_retrieve_interviews_query_augmentation(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test that interview query is augmented with skills."""
        captured_vectors = []

        original_encode = mock_embedding_service.encode

        def capture_encode(text):
            captured_vectors.append(text)
            return original_encode(text)

        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            mock_embedding_service, "encode", capture_encode
        )

        retrieve_interviews("Python developer", sample_requirements)

        # The query should be augmented with interview_required_skills
        assert len(captured_vectors) > 0
        # At least one call should contain the augmented query
        assert any("Skills:" in v for v in captured_vectors)

    def test_retrieve_interviews_empty_results(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test handling of empty interview results."""
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: [],
        )

        results = retrieve_interviews("Python", sample_requirements)
        assert results == []


# ===========================================================================
# Candidate Aggregation Tests
# ===========================================================================


class TestCandidateAggregation:
    """Tests for candidate aggregation from multiple result sets."""

    def test_aggregate_profiles_and_interviews(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test aggregating profile and interview results."""
        results = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )

        assert len(results) > 0
        # Should have candidates from both sources
        candidate_ids = {r.candidate_id for r in results}
        assert "CAND_001" in candidate_ids
        assert "CAND_002" in candidate_ids

    def test_aggregate_profiles_only(
        self, sample_profile_results, sample_requirements
    ):
        """Test aggregation with only profile results."""
        results = rank_candidates(
            sample_profile_results, [], sample_requirements
        )

        assert len(results) == 3
        # All should have interview_score of 0
        for r in results:
            assert r.interview_score == 0.0

    def test_aggregate_interviews_only(
        self, sample_interview_results, sample_requirements
    ):
        """Test aggregation with only interview results."""
        results = rank_candidates(
            [], sample_interview_results, sample_requirements
        )

        assert len(results) > 0
        # All should have profile_score of 0
        for r in results:
            assert r.profile_score == 0.0

    def test_aggregate_empty(self, sample_requirements):
        """Test aggregation with no results."""
        results = rank_candidates([], [], sample_requirements)
        assert results == []

    def test_aggregate_deterministic(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test that aggregation is deterministic (same input = same output)."""
        results1 = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )
        results2 = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.candidate_id == r2.candidate_id
            assert r1.relevance_score == r2.relevance_score


# ===========================================================================
# Scoring Formula Tests
# ===========================================================================


class TestScoringFormula:
    """Tests for the scoring formula and weights."""

    def test_scoring_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        from app.core.config import settings

        total = (
            settings.PROFILE_WEIGHT
            + settings.INTERVIEW_WEIGHT
            + settings.STRUCTURED_WEIGHT
        )
        assert abs(total - 1.0) < 0.001

    def test_final_score_calculation(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test that final score is correctly calculated from components."""
        from app.core.config import settings

        results = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )

        for r in results:
            expected = (
                settings.PROFILE_WEIGHT * r.profile_score
                + settings.INTERVIEW_WEIGHT * r.interview_score
                + settings.STRUCTURED_WEIGHT * r.structured_score
            )
            assert abs(r.relevance_score - expected) < 0.001

    def test_profile_score_is_max(
        self, sample_requirements
    ):
        """Test that profile score is the maximum across all profile hits."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.7,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert len(results) == 1
        assert results[0].profile_score == 0.9

    def test_interview_score_is_max(self, sample_requirements):
        """Test that interview score is the maximum across all interview hits."""
        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.6,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 1"},
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.85,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 2"},
            },
        ]

        results = rank_candidates([], interview_results, sample_requirements)

        assert len(results) == 1
        assert results[0].interview_score == 0.85

    def test_score_normalization(self, sample_requirements):
        """Test that scores are normalized to [0, 1]."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 1.5,  # Above 1.0
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)
        assert results[0].profile_score <= 1.0

    def test_score_normalization_negative(self, sample_requirements):
        """Test that negative scores are clamped to 0."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": -0.5,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)
        assert results[0].profile_score >= 0.0


# ===========================================================================
# Structured Filtering Tests
# ===========================================================================


class TestStructuredFiltering:
    """Tests for structured requirement matching."""

    def test_skills_match(self, sample_requirements):
        """Test that skills matching works correctly."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert "python" in results[0].matched_skills
        assert "fastapi" in results[0].matched_skills
        assert "rag" in results[0].matched_skills
        assert len(results[0].missing_requirements) == 0

    def test_skills_partial_match(self, sample_requirements):
        """Test partial skills matching."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],  # Only 1 of 3 required
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert "python" in results[0].matched_skills
        assert len(results[0].missing_requirements) > 0
        assert any("fastapi" in m for m in results[0].missing_requirements)

    def test_experience_match(self, sample_requirements):
        """Test experience range matching."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,  # Within 3-8 range
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert not any("Experience" in m for m in results[0].missing_requirements)

    def test_experience_mismatch(self, sample_requirements):
        """Test experience outside range."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 2.0,  # Below 3 minimum
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert any("Experience" in m for m in results[0].missing_requirements)

    def test_location_match(self, sample_requirements):
        """Test location matching."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",  # Matches
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert not any("Location" in m for m in results[0].missing_requirements)

    def test_location_mismatch(self, sample_requirements):
        """Test location mismatch."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Mumbai",  # Doesn't match Bangalore
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert any("Location" in m for m in results[0].missing_requirements)


# ===========================================================================
# Full Search Pipeline Tests
# ===========================================================================


class TestSearchPipeline:
    """Tests for the end-to-end search pipeline."""

    def test_search_pipeline_basic(
        self,
        monkeypatch,
        mock_embedding_service,
        sample_profile_results,
        sample_interview_results,
        sample_requirements,
    ):
        """Test the full search pipeline."""
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: sample_profile_results,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: sample_interview_results,
        )
        # Mock the summarizer to avoid LLM calls
        monkeypatch.setattr(
            "app.retrieval.candidate_ranker.generate_candidate_summary",
            lambda r: f"Summary for {r.name}",
        )

        results = search_pipeline("Python FastAPI RAG", sample_requirements)

        assert len(results) > 0
        assert all(r.summary != "" for r in results)
        # Results should be sorted by relevance
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score

    def test_search_pipeline_no_results(
        self, monkeypatch, mock_embedding_service, sample_requirements
    ):
        """Test pipeline with no matching candidates."""
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: [],
        )

        results = search_pipeline("Python FastAPI RAG", sample_requirements)
        assert results == []

    def test_search_pipeline_interview_fallback(
        self,
        monkeypatch,
        mock_embedding_service,
        sample_interview_results,
        sample_requirements,
    ):
        """Test that interview search works when no profiles found."""
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: sample_interview_results,
        )
        monkeypatch.setattr(
            "app.retrieval.candidate_ranker.generate_candidate_summary",
            lambda r: f"Summary for {r.candidate_id}",
        )

        results = search_pipeline("Python FastAPI RAG", sample_requirements)

        # Should still find candidates from interviews
        assert len(results) > 0

    def test_search_pipeline_result_count(
        self,
        monkeypatch,
        mock_embedding_service,
        sample_profile_results,
        sample_interview_results,
        sample_requirements,
    ):
        """Test that pipeline respects FINAL_TOP_K limit."""
        from app.core.config import settings

        monkeypatch.setattr(
            "app.retrieval.profile_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.embedding_service",
            mock_embedding_service,
        )
        monkeypatch.setattr(
            "app.retrieval.profile_retriever.search_profiles",
            lambda **kwargs: sample_profile_results,
        )
        monkeypatch.setattr(
            "app.retrieval.interview_retriever.search_interviews",
            lambda **kwargs: sample_interview_results,
        )
        monkeypatch.setattr(
            "app.retrieval.candidate_ranker.generate_candidate_summary",
            lambda r: "",
        )

        results = search_pipeline("Python FastAPI RAG", sample_requirements)

        assert len(results) <= settings.FINAL_TOP_K


# ===========================================================================
# Deterministic Ranking Tests
# ===========================================================================


class TestDeterministicRanking:
    """Tests to verify deterministic ranking behavior."""

    def test_ranking_order_is_deterministic(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test that ranking is deterministic across multiple runs."""
        all_results = []
        for _ in range(5):
            results = rank_candidates(
                sample_profile_results, sample_interview_results, sample_requirements
            )
            all_results.append([(r.candidate_id, r.relevance_score) for r in results])

        # All runs should produce identical results
        for i in range(1, len(all_results)):
            assert all_results[0] == all_results[i]

    def test_ranking_respects_score_order(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test that results are sorted by relevance score descending."""
        results = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )

        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score

    def test_ranking_assigns_sequential_ranks(
        self, sample_profile_results, sample_interview_results, sample_requirements
    ):
        """Test that ranks are assigned sequentially starting from 1."""
        results = rank_candidates(
            sample_profile_results, sample_interview_results, sample_requirements
        )

        for i, r in enumerate(results, start=1):
            assert r.rank == i
