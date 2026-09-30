"""Tests specifically for candidate ranking logic."""

import pytest

from app.core.config import settings
from app.retrieval.candidate_ranker import (
    _calculate_structured_score,
    _normalize_score,
    rank_candidates,
)
from app.schemas.search import InterpretedRequirements


# ===========================================================================
# Score Normalization Tests
# ===========================================================================


class TestNormalizeScore:
    """Tests for score normalization."""

    def test_normalize_within_range(self):
        """Test that scores within [0, 1] are unchanged."""
        assert _normalize_score(0.5) == 0.5
        assert _normalize_score(0.0) == 0.0
        assert _normalize_score(1.0) == 1.0

    def test_normalize_above_range(self):
        """Test that scores above 1.0 are clamped."""
        assert _normalize_score(1.5) == 1.0
        assert _normalize_score(2.0) == 1.0
        assert _normalize_score(100.0) == 1.0

    def test_normalize_below_range(self):
        """Test that negative scores are clamped to 0."""
        assert _normalize_score(-0.5) == 0.0
        assert _normalize_score(-1.0) == 0.0
        assert _normalize_score(-100.0) == 0.0

    def test_normalize_edge_cases(self):
        """Test edge cases."""
        assert _normalize_score(0.0) == 0.0
        assert _normalize_score(1.0) == 1.0
        assert _normalize_score(0.999) == 0.999
        assert _normalize_score(0.001) == 0.001

    def test_normalize_preserves_float(self):
        """Test that normalization preserves float type."""
        result = _normalize_score(0.123456)
        assert isinstance(result, float)


# ===========================================================================
# Weight Configuration Tests
# ===========================================================================


class TestWeightConfiguration:
    """Tests for ranking weight configuration."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        total = (
            settings.PROFILE_WEIGHT
            + settings.INTERVIEW_WEIGHT
            + settings.STRUCTURED_WEIGHT
        )
        assert abs(total - 1.0) < 0.001

    def test_profile_weight_is_dominant(self):
        """Test that profile weight is the largest component."""
        assert settings.PROFILE_WEIGHT >= settings.INTERVIEW_WEIGHT
        assert settings.PROFILE_WEIGHT >= settings.STRUCTURED_WEIGHT

    def test_interview_weight_is_second(self):
        """Test that interview weight is the second largest."""
        assert settings.INTERVIEW_WEIGHT >= settings.STRUCTURED_WEIGHT

    def test_weights_are_positive(self):
        """Test that all weights are positive."""
        assert settings.PROFILE_WEIGHT > 0
        assert settings.INTERVIEW_WEIGHT > 0
        assert settings.STRUCTURED_WEIGHT > 0

    def test_weights_are_reasonable(self):
        """Test that weights are within reasonable bounds."""
        assert 0.0 < settings.PROFILE_WEIGHT < 1.0
        assert 0.0 < settings.INTERVIEW_WEIGHT < 1.0
        assert 0.0 < settings.STRUCTURED_WEIGHT < 1.0


# ===========================================================================
# Candidate Aggregation by ID Tests
# ===========================================================================


class TestCandidateAggregation:
    """Tests for aggregating multiple results by candidate ID."""

    def test_aggregate_multiple_profile_hits(self, sample_requirements):
        """Test aggregating multiple profile hits for same candidate."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.7,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert len(results) == 1
        # Should use the highest score
        assert results[0].profile_score == 0.9

    def test_aggregate_multiple_interview_hits(self, sample_requirements):
        """Test aggregating multiple interview hits for same candidate."""
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
            {
                "candidate_id": "CAND_001",
                "score": 0.75,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 3"},
            },
        ]

        results = rank_candidates([], interview_results, sample_requirements)

        assert len(results) == 1
        assert results[0].interview_score == 0.85

    def test_aggregate_union_of_candidates(self, sample_requirements):
        """Test that union of profile and interview candidates is used."""
        profile_results = [
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
            {
                "candidate_id": "CAND_002",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_002",
                    "skills": ["python"],
                    "experience": 4.0,
                    "location": "Mumbai",
                },
            },
        ]

        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.85,
                "payload": {"candidate_id": "CAND_001", "text": "chunk"},
            },
            {
                "candidate_id": "CAND_003",
                "score": 0.7,
                "payload": {"candidate_id": "CAND_003", "text": "chunk"},
            },
        ]

        results = rank_candidates(profile_results, interview_results, sample_requirements)

        candidate_ids = {r.candidate_id for r in results}
        assert candidate_ids == {"CAND_001", "CAND_002", "CAND_003"}

    def test_aggregate_preserves_all_evidence(self, sample_requirements):
        """Test that all evidence is preserved during aggregation."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                    "source_file": "resume.pdf",
                },
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                    "source_file": "resume.pdf",
                },
            },
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        # Should have evidence from both profile hits
        assert len(results[0].profile_evidence) == 2


# ===========================================================================
# Evidence Collection Tests
# ===========================================================================


class TestEvidenceCollection:
    """Tests for evidence collection during ranking."""

    def test_profile_evidence_collected(self, sample_requirements):
        """Test that profile evidence is collected."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                    "source_file": "CAND_001_resume.txt",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert len(results[0].profile_evidence) == 1
        assert results[0].profile_evidence[0].source == "profile"
        assert results[0].profile_evidence[0].source_file == "CAND_001_resume.txt"

    def test_interview_evidence_collected(self, sample_requirements):
        """Test that interview evidence is collected."""
        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.85,
                "payload": {
                    "candidate_id": "CAND_001",
                    "text": "I built a RAG pipeline",
                    "source_file": "CAND_001_interview.txt",
                    "timestamp": "00:05:30",
                    "evidence_type": "demonstrated",
                },
            }
        ]

        results = rank_candidates([], interview_results, sample_requirements)

        assert len(results[0].interview_evidence) == 1
        assert results[0].interview_evidence[0].source == "interview"
        assert results[0].interview_evidence[0].text == "I built a RAG pipeline"
        assert results[0].interview_evidence[0].evidence_type == "demonstrated"

    def test_source_documents_collected(self, sample_requirements):
        """Test that source documents are collected from evidence."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                    "source_file": "resume.txt",
                },
            }
        ]

        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.85,
                "payload": {
                    "candidate_id": "CAND_001",
                    "text": "chunk",
                    "source_file": "interview.txt",
                },
            }
        ]

        results = rank_candidates(profile_results, interview_results, sample_requirements)

        source_docs = set(results[0].source_documents)
        assert "resume.txt" in source_docs
        assert "interview.txt" in source_docs

    def test_no_evidence_when_no_results(self, sample_requirements):
        """Test that no evidence is collected when there are no results."""
        results = rank_candidates([], [], sample_requirements)
        assert results == []


# ===========================================================================
# Missing Requirements Detection Tests
# ===========================================================================


class TestMissingRequirements:
    """Tests for detecting missing requirements."""

    def test_missing_skills_detected(self, sample_requirements):
        """Test that missing skills are detected."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],  # Missing fastapi, rag
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        missing = results[0].missing_requirements
        assert any("fastapi" in m for m in missing)
        assert any("rag" in m for m in missing)

    def test_missing_experience_detected(self, sample_requirements):
        """Test that experience mismatch is detected."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 2.0,  # Below minimum of 3
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert any("Experience" in m for m in results[0].missing_requirements)

    def test_missing_location_detected(self, sample_requirements):
        """Test that location mismatch is detected."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Delhi",  # Not Bangalore
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert any("Location" in m for m in results[0].missing_requirements)

    def test_no_missing_when_all_match(self, sample_requirements):
        """Test that no missing requirements when all match."""
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

        assert len(results[0].missing_requirements) == 0

    def test_missing_requirements_empty_when_no_constraints(
        self, sample_requirements_no_constraints
    ):
        """Test that no missing requirements when no constraints specified."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["java"],
                    "experience": 1.0,
                    "location": "Delhi",
                },
            }
        ]

        results = rank_candidates(
            profile_results, [], sample_requirements_no_constraints
        )

        assert len(results[0].missing_requirements) == 0


# ===========================================================================
# Interview Evidence Boost Tests
# ===========================================================================


class TestInterviewEvidenceBoost:
    """Tests for interview evidence boosting candidate scores."""

    def test_interview_evidence_improves_score(self, sample_requirements):
        """Test that having interview evidence improves the final score."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        # Without interview evidence
        results_without = rank_candidates(profile_results, [], sample_requirements)

        # With interview evidence
        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {"candidate_id": "CAND_001", "text": "chunk"},
            }
        ]
        results_with = rank_candidates(
            profile_results, interview_results, sample_requirements
        )

        assert results_with[0].relevance_score > results_without[0].relevance_score

    def test_candidates_with_interviews_rank_higher(self, sample_requirements):
        """Test that candidates with interview evidence rank higher than those without."""
        # Two candidates with identical profiles
        profile_results = [
            {
                "candidate_id": "CAND_WITH",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_WITH",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
            {
                "candidate_id": "CAND_WITHOUT",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_WITHOUT",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            },
        ]

        # Only one has interview evidence
        interview_results = [
            {
                "candidate_id": "CAND_WITH",
                "score": 0.85,
                "payload": {"candidate_id": "CAND_WITH", "text": "chunk"},
            }
        ]

        results = rank_candidates(profile_results, interview_results, sample_requirements)

        # Find both candidates
        cand_with = next(r for r in results if r.candidate_id == "CAND_WITH")
        cand_without = next(r for r in results if r.candidate_id == "CAND_WITHOUT")

        # Candidate with interview evidence should rank higher
        assert cand_with.relevance_score > cand_without.relevance_score
        assert cand_with.rank < cand_without.rank

    def test_interview_score_zero_when_no_evidence(self, sample_requirements):
        """Test that interview score is 0 when no interview evidence exists."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results = rank_candidates(profile_results, [], sample_requirements)

        assert results[0].interview_score == 0.0
        assert results[0].interview_evidence == []

    def test_interview_boost_weight(self, sample_requirements):
        """Test that interview boost is proportional to INTERVIEW_WEIGHT."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        results_without = rank_candidates(profile_results, [], sample_requirements)

        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 1.0,  # Maximum interview score
                "payload": {"candidate_id": "CAND_001", "text": "chunk"},
            }
        ]
        results_with = rank_candidates(
            profile_results, interview_results, sample_requirements
        )

        # The difference should be INTERVIEW_WEIGHT * 1.0
        score_diff = results_with[0].relevance_score - results_without[0].relevance_score
        assert abs(score_diff - settings.INTERVIEW_WEIGHT * 1.0) < 0.001

    def test_multiple_interview_chunks_use_max(self, sample_requirements):
        """Test that multiple interview chunks use the maximum score."""
        profile_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.8,
                "payload": {
                    "candidate_id": "CAND_001",
                    "skills": ["python", "fastapi", "rag"],
                    "experience": 5.0,
                    "location": "Bangalore",
                },
            }
        ]

        interview_results = [
            {
                "candidate_id": "CAND_001",
                "score": 0.5,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 1"},
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.9,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 2"},
            },
            {
                "candidate_id": "CAND_001",
                "score": 0.7,
                "payload": {"candidate_id": "CAND_001", "text": "chunk 3"},
            },
        ]

        results = rank_candidates(
            profile_results, interview_results, sample_requirements
        )

        # Should use the maximum interview score (0.9)
        assert results[0].interview_score == 0.9
        # But should have evidence from all chunks
        assert len(results[0].interview_evidence) == 3


# ===========================================================================
# Structured Score Calculation Tests
# ===========================================================================


class TestStructuredScoreCalculation:
    """Tests for the structured score calculation."""

    def test_structured_score_perfect_match(self, sample_requirements):
        """Test structured score when all requirements match."""
        payload = {
            "skills": ["python", "fastapi", "rag"],
            "experience": 5.0,
            "location": "Bangalore",
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        assert score == 1.0
        assert len(matched) == 3
        assert len(missing) == 0

    def test_structured_score_no_match(self, sample_requirements):
        """Test structured score when no requirements match."""
        payload = {
            "skills": ["java", "spring"],
            "experience": 1.0,
            "location": "Delhi",
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        assert score == 0.0
        assert len(matched) == 0
        assert len(missing) > 0

    def test_structured_score_partial_skills(self, sample_requirements):
        """Test structured score with partial skills match."""
        payload = {
            "skills": ["python", "fastapi"],  # Missing rag
            "experience": 5.0,
            "location": "Bangalore",
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        # Skills: 2/3 = 0.667, Experience: 1.0, Location: 1.0
        # Weighted: 0.5 * 0.667 + 0.3 * 1.0 + 0.2 * 1.0 = 0.833
        expected = 0.5 * (2 / 3) + 0.3 * 1.0 + 0.2 * 1.0
        assert abs(score - expected) < 0.001

    def test_structured_score_no_requirements(self, sample_requirements_no_constraints):
        """Test structured score when no requirements specified."""
        payload = {
            "skills": ["java"],
            "experience": 1.0,
            "location": "Delhi",
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements_no_constraints
        )

        # When no requirements, score should be 1.0 (no penalty)
        assert score == 1.0
        assert len(missing) == 0

    def test_structured_score_skills_weight(self, sample_requirements):
        """Test that skills contribute 50% to structured score."""
        # Only skills match, experience and location don't
        payload = {
            "skills": ["python", "fastapi", "rag"],
            "experience": 1.0,  # Below minimum
            "location": "Delhi",  # Wrong location
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        # Skills: 1.0, Experience: 0.0, Location: 0.0
        # Weighted: 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.0 = 0.5
        assert abs(score - 0.5) < 0.001

    def test_structured_score_experience_weight(self, sample_requirements):
        """Test that experience contributes 30% to structured score."""
        # Only experience matches
        payload = {
            "skills": ["java"],  # Wrong skills
            "experience": 5.0,  # Within range
            "location": "Delhi",  # Wrong location
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        # Skills: 0.0, Experience: 1.0, Location: 0.0
        # Weighted: 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.0 = 0.3
        assert abs(score - 0.3) < 0.001

    def test_structured_score_location_weight(self, sample_requirements):
        """Test that location contributes 20% to structured score."""
        # Only location matches
        payload = {
            "skills": ["java"],  # Wrong skills
            "experience": 1.0,  # Below minimum
            "location": "Bangalore",  # Correct location
        }

        score, matched, missing = _calculate_structured_score(
            payload, sample_requirements
        )

        # Skills: 0.0, Experience: 0.0, Location: 1.0
        # Weighted: 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0 = 0.2
        assert abs(score - 0.2) < 0.001
