"""Tests for document ingestion: loaders, normalization, parsing, and chunking."""

import io

import pytest

from app.ingestion.chunker import (
    Chunk,
    chunk_interview_transcript,
    chunk_text,
    split_by_speaker,
    split_into_sentences,
)
from app.ingestion.loaders import (
    detect_file_type,
    load_docx,
    load_document,
    load_pdf,
    load_srt,
    load_txt,
    load_vtt,
)
from app.ingestion.parser import (
    extract_companies,
    extract_education,
    extract_email,
    extract_experience_years,
    extract_name,
    extract_phone,
    extract_skills,
    normalize_text,
    parse_candidate_profile,
)


# ===========================================================================
# PDF Loading Tests
# ===========================================================================


class TestPDFLoader:
    """Tests for PDF document loading."""

    def test_load_pdf_with_mock(self, monkeypatch):
        """Test PDF loading with a mocked pdfplumber."""
        import pdfplumber

        class MockPage:
            def extract_text(self):
                return "Hello from PDF page"

        class MockPDF:
            def __init__(self, *args, **kwargs):
                self.pages = [MockPage(), MockPage()]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(pdfplumber, "open", lambda *a, **kw: MockPDF())

        result = load_pdf(b"fake pdf bytes")
        assert "Hello from PDF page" in result
        assert result.count("Hello from PDF page") == 2

    def test_load_pdf_empty_pages(self, monkeypatch):
        """Test PDF loading when pages have no text."""
        import pdfplumber

        class MockPage:
            def extract_text(self):
                return None

        class MockPDF:
            def __init__(self, *args, **kwargs):
                self.pages = [MockPage()]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(pdfplumber, "open", lambda *a, **kw: MockPDF())

        result = load_pdf(b"fake pdf bytes")
        assert result == ""

    def test_load_pdf_raises_on_error(self, monkeypatch):
        """Test that PDF loading raises on invalid data."""
        import pdfplumber

        def mock_open(*args, **kwargs):
            raise ValueError("Invalid PDF")

        monkeypatch.setattr(pdfplumber, "open", mock_open)

        with pytest.raises(ValueError, match="Invalid PDF"):
            load_pdf(b"not a real pdf")


# ===========================================================================
# DOCX Loading Tests
# ===========================================================================


class TestDOCXLoader:
    """Tests for DOCX document loading."""

    def test_load_docx_with_mock(self, monkeypatch):
        """Test DOCX loading with a mocked python-docx Document."""
        from docx import Document as RealDocument

        class MockCell:
            def __init__(self, text):
                self.text = text

        class MockRow:
            def __init__(self, cells):
                self.cells = [MockCell(c) for c in cells]

        class MockTable:
            def __init__(self):
                self.rows = [MockRow(["Skill", "Level"]), MockRow(["Python", "Expert"])]

        class MockParagraph:
            def __init__(self, text):
                self.text = text

        class MockDoc:
            def __init__(self, *args, **kwargs):
                self.paragraphs = [
                    MockParagraph("John Doe"),
                    MockParagraph("Software Engineer"),
                    MockParagraph(""),
                ]
                self.tables = [MockTable()]

        monkeypatch.setattr("app.ingestion.loaders.Document", MockDoc)

        result = load_docx(b"fake docx bytes")
        assert "John Doe" in result
        assert "Software Engineer" in result
        assert "Skill | Level" in result
        assert "Python | Expert" in result

    def test_load_docx_no_tables(self, monkeypatch):
        """Test DOCX loading with paragraphs only."""
        class MockParagraph:
            def __init__(self, text):
                self.text = text

        class MockDoc:
            def __init__(self, *args, **kwargs):
                self.paragraphs = [MockParagraph("Line 1"), MockParagraph("Line 2")]
                self.tables = []

        monkeypatch.setattr("app.ingestion.loaders.Document", MockDoc)

        result = load_docx(b"fake docx bytes")
        assert "Line 1" in result
        assert "Line 2" in result


# ===========================================================================
# TXT Loading Tests
# ===========================================================================


class TestTXTLoader:
    """Tests for plain text loading."""

    def test_load_txt_basic(self):
        """Test basic text loading."""
        content = b"Hello, this is a test document.\nWith multiple lines."
        result = load_txt(content)
        assert result == "Hello, this is a test document.\nWith multiple lines."

    def test_load_txt_utf8(self):
        """Test UTF-8 text loading."""
        content = "Unicode: café, naïve, 日本語".encode("utf-8")
        result = load_txt(content)
        assert "café" in result
        assert "日本語" in result

    def test_load_txt_invalid_bytes(self):
        """Test that invalid bytes are replaced, not raised."""
        content = b"Hello \xff\xfe World"
        result = load_txt(content)
        assert "Hello" in result
        assert "World" in result

    def test_load_txt_empty(self):
        """Test empty text file."""
        result = load_txt(b"")
        assert result == ""


# ===========================================================================
# SRT/VTT Loading Tests
# ===========================================================================


class TestSRTLoader:
    """Tests for SRT subtitle loading."""

    def test_load_srt_basic(self):
        """Test basic SRT loading."""
        content = b"""1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
This is a test
"""
        result = load_srt(content)
        assert "[00:00:01,000 --> 00:00:04,000]" in result
        assert "Hello world" in result
        assert "This is a test" in result

    def test_load_srt_multiline(self):
        """Test SRT with multi-line subtitles."""
        content = b"""1
00:00:01,000 --> 00:00:04,000
Line one
Line two
Line three
"""
        result = load_srt(content)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result

    def test_load_srt_empty(self):
        """Test empty SRT file."""
        result = load_srt(b"")
        assert result == ""


class TestVTTLoader:
    """Tests for VTT subtitle loading."""

    def test_load_vtt_basic(self):
        """Test basic VTT loading."""
        content = b"""WEBVTT

00:00:01.000 --> 00:00:04.000
Hello from VTT

00:00:05.000 --> 00:00:08.000
Second subtitle
"""
        result = load_vtt(content)
        assert "Hello from VTT" in result
        assert "Second subtitle" in result
        assert "WEBVTT" not in result

    def test_load_vtt_with_notes(self):
        """Test VTT with NOTE blocks."""
        content = b"""WEBVTT

NOTE This is a comment
that should be skipped

00:00:01.000 --> 00:00:04.000
Actual subtitle
"""
        result = load_vtt(content)
        assert "Actual subtitle" in result
        assert "NOTE" not in result

    def test_load_vtt_empty(self):
        """Test empty VTT file."""
        result = load_vtt(b"")
        assert result == ""


# ===========================================================================
# File Type Detection Tests
# ===========================================================================


class TestFileTypeDetection:
    """Tests for file type detection."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("resume.pdf", "pdf"),
            ("resume.docx", "docx"),
            ("resume.txt", "txt"),
            ("resume.md", "md"),
            ("subtitles.srt", "srt"),
            ("subtitles.vtt", "vtt"),
            ("resume.PDF", "pdf"),
            ("resume.DOCX", "docx"),
            ("archive.zip", "unknown"),
            ("noextension", "unknown"),
        ],
    )
    def test_detect_file_type(self, filename, expected):
        """Test file type detection from extension."""
        assert detect_file_type(filename) == expected


# ===========================================================================
# Document Loading Integration Tests
# ===========================================================================


class TestLoadDocument:
    """Tests for the main load_document function."""

    def test_load_document_txt(self):
        """Test loading a TXT document."""
        text, file_type = load_document(b"Hello world", "test.txt")
        assert text == "Hello world"
        assert file_type == "txt"

    def test_load_document_unsupported(self):
        """Test that unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(b"data", "file.xyz")

    def test_load_document_md_as_txt(self):
        """Test that .md files are loaded as text."""
        text, file_type = load_document(b"# Markdown", "readme.md")
        assert text == "# Markdown"
        assert file_type == "md"


# ===========================================================================
# Text Normalization Tests
# ===========================================================================


class TestNormalizeText:
    """Tests for text normalization."""

    def test_normalize_line_endings(self):
        """Test that line endings are normalized."""
        text = "Line 1\r\nLine 2\rLine 3\n"
        result = normalize_text(text)
        assert "\r\n" not in result
        assert "\r" not in result

    def test_normalize_excessive_blank_lines(self):
        """Test that excessive blank lines are collapsed."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = normalize_text(text)
        assert "\n\n\n" not in result

    def test_normalize_excessive_spaces(self):
        """Test that multiple spaces are collapsed."""
        text = "Word1    Word2     Word3"
        result = normalize_text(text)
        assert "  " not in result

    def test_normalize_removes_control_chars(self):
        """Test that control characters are removed."""
        text = "Hello\x00World\x07Test"
        result = normalize_text(text)
        assert "\x00" not in result
        assert "\x07" not in result

    def test_normalize_removes_punctuation_only_lines(self):
        """Test that lines with only punctuation are removed."""
        text = "Real line\n!@#$%^&*()\nAnother real line"
        result = normalize_text(text)
        assert "!@#$%^&*()" not in result
        assert "Real line" in result
        assert "Another real line" in result

    def test_normalize_preserves_meaningful_content(self):
        """Test that meaningful content is preserved."""
        text = "Name: John Doe\nExperience: 5 years"
        result = normalize_text(text)
        assert "Name: John Doe" in result
        assert "Experience: 5 years" in result

    def test_normalize_empty_string(self):
        """Test normalizing an empty string."""
        assert normalize_text("") == ""


# ===========================================================================
# Profile Extraction Tests
# ===========================================================================


class TestExtractName:
    """Tests for name extraction."""

    def test_extract_name_from_label(self):
        """Test extracting name from 'Name:' label."""
        text = "Name: John Doe\nEmail: john@example.com"
        assert extract_name(text) == "John Doe"

    def test_extract_name_from_first_line(self):
        """Test extracting name from first line."""
        text = "Jane Smith\nSoftware Engineer\n5 years experience"
        assert extract_name(text) == "Jane Smith"

    def test_extract_name_from_resume_of(self):
        """Test extracting name from 'Resume of X' pattern."""
        text = "Resume of John Doe\nExperience: 5 years"
        assert extract_name(text) == "John Doe"

    def test_extract_name_skips_headers(self):
        """Test that common headers are skipped."""
        text = "RESUME\nJohn Doe\nDeveloper"
        assert extract_name(text) == "John Doe"

    def test_extract_name_empty(self):
        """Test with no name found."""
        text = "12345\n67890"
        assert extract_name(text) == ""


class TestExtractEmail:
    """Tests for email extraction."""

    def test_extract_email_standard(self):
        """Test extracting a standard email."""
        text = "Contact: john.doe@example.com"
        assert extract_email(text) == "john.doe@example.com"

    def test_extract_email_with_plus(self):
        """Test extracting email with plus addressing."""
        text = "Email: user+tag@domain.co.uk"
        assert extract_email(text) == "user+tag@domain.co.uk"

    def test_extract_email_none_found(self):
        """Test when no email is present."""
        text = "No email here"
        assert extract_email(text) == ""


class TestExtractPhone:
    """Tests for phone number extraction."""

    def test_extract_phone_international(self):
        """Test extracting international format."""
        text = "Phone: +91-98765-43210"
        result = extract_phone(text)
        assert "98765" in result

    def test_extract_phone_us_format(self):
        """Test extracting US format."""
        text = "Call: (555) 123-4567"
        result = extract_phone(text)
        assert "555" in result

    def test_extract_phone_none_found(self):
        """Test when no phone is present."""
        text = "No phone number"
        assert extract_phone(text) == ""


class TestExtractExperience:
    """Tests for experience years extraction."""

    def test_extract_experience_standard(self):
        """Test standard 'X years of experience' format."""
        text = "I have 5 years of experience in Python"
        assert extract_experience_years(text) == 5.0

    def test_extract_experience_plus(self):
        """Test 'X+ years' format."""
        text = "Over 7+ years of experience"
        assert extract_experience_years(text) == 7.0

    def test_extract_experience_multiple(self):
        """Test that maximum is returned when multiple mentions exist."""
        text = "3 years at Company A, 5 years at Company B, total 8 years"
        assert extract_experience_years(text) == 8.0

    def test_extract_experience_none(self):
        """Test when no experience is mentioned."""
        text = "Recent graduate"
        assert extract_experience_years(text) == 0.0

    def test_extract_experience_unrealistic(self):
        """Test that unrealistic values are filtered."""
        text = "I have 100 years of experience"
        assert extract_experience_years(text) == 0.0


class TestExtractSkills:
    """Tests for skills extraction."""

    def test_extract_skills_known(self):
        """Test extracting known skills."""
        text = "I know Python, FastAPI, and Docker"
        skills = extract_skills(text)
        assert "python" in skills
        assert "fastapi" in skills
        assert "docker" in skills

    def test_extract_skills_case_insensitive(self):
        """Test that skill extraction is case-insensitive."""
        text = "I use PYTHON and FastAPI"
        skills = extract_skills(text)
        assert "python" in skills
        assert "fastapi" in skills

    def test_extract_skills_none_found(self):
        """Test when no known skills are present."""
        text = "I enjoy hiking and reading"
        skills = extract_skills(text)
        assert len(skills) == 0

    def test_extract_skills_sorted(self):
        """Test that skills are returned sorted."""
        text = "Python, Docker, AWS, Git"
        skills = extract_skills(text)
        assert skills == sorted(skills)


class TestExtractEducation:
    """Tests for education extraction."""

    def test_extract_education_btech(self):
        """Test extracting B.Tech education."""
        text = "B.Tech in Computer Science from IIT Delhi"
        result = extract_education(text)
        assert "b.tech" in result.lower() or "computer science" in result.lower()

    def test_extract_education_masters(self):
        """Test extracting Master's degree."""
        text = "M.S. in Data Science from Stanford"
        result = extract_education(text)
        assert "data science" in result.lower() or "stanford" in result.lower()

    def test_extract_education_none(self):
        """Test when no education info is present."""
        text = "Work experience only"
        result = extract_education(text)
        assert result == ""


class TestExtractCompanies:
    """Tests for company name extraction."""

    def test_extract_companies_at_pattern(self):
        """Test extracting companies from 'at Company' pattern."""
        text = "Worked at Google (2020-2023) and at Microsoft (2018-2020)"
        companies = extract_companies(text)
        assert len(companies) > 0

    def test_extract_companies_none(self):
        """Test when no companies are found."""
        text = "Freelance developer"
        companies = extract_companies(text)
        assert len(companies) == 0


# ===========================================================================
# Parse Candidate Profile Tests
# ===========================================================================


class TestParseCandidateProfile:
    """Tests for the full profile parsing pipeline."""

    def test_parse_complete_profile(self, sample_resume_text):
        """Test parsing a complete resume."""
        profile = parse_candidate_profile(sample_resume_text, "test.txt")

        assert profile["name"] == "Arjun Kumar"
        assert profile["email"] == "arjun.kumar@email.com"
        assert "9876543210" in profile["phone"]
        assert profile["total_experience"] == 5.0
        assert "python" in profile["skills"]
        assert "fastapi" in profile["skills"]
        assert profile["education"] != ""
        assert profile["profile_source"] == "test.txt"
        assert profile["resume_text"] == sample_resume_text

    def test_parse_profile_minimal(self):
        """Test parsing a minimal resume."""
        text = "John Doe\nPython developer"
        profile = parse_candidate_profile(text, "minimal.txt")

        assert profile["name"] == "John Doe"
        assert profile["email"] == ""
        assert profile["phone"] == ""
        assert profile["total_experience"] == 0.0
        assert "python" in profile["skills"]

    def test_parse_profile_empty(self):
        """Test parsing an empty resume."""
        profile = parse_candidate_profile("", "empty.txt")

        assert profile["name"] == ""
        assert profile["email"] == ""
        assert profile["total_experience"] == 0.0
        assert profile["skills"] == []


# ===========================================================================
# Chunking Tests
# ===========================================================================


class TestChunkText:
    """Tests for text chunking."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].index == 0

    def test_chunk_text_correct_number(self):
        """Test that chunking produces the expected number of chunks."""
        # Create text with known sentence structure
        sentences = ["This is a test sentence."] * 20
        text = " ".join(sentences)

        chunks = chunk_text(text, chunk_size=200, overlap=50)

        assert len(chunks) > 1
        # Verify indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        sentences = ["Sentence number {}.".format(i) for i in range(30)]
        text = " ".join(sentences)

        chunks = chunk_text(text, chunk_size=150, overlap=50)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            # (overlap means some sentences appear in both)
            chunk1_words = set(chunks[0].text.split())
            chunk2_words = set(chunks[1].text.split())
            # There should be some overlap in content
            assert len(chunk1_words & chunk2_words) > 0

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunks = chunk_text("   \n\n   ", chunk_size=100, overlap=20)
        assert chunks == []

    def test_chunk_text_single_sentence(self):
        """Test chunking a single sentence."""
        text = "This is one single sentence."
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_text_char_positions(self):
        """Test that character positions are tracked."""
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        for chunk in chunks:
            assert chunk.start_char < chunk.end_char
            assert chunk.end_char - chunk.start_char == len(chunk.text)


class TestChunkInterviewTranscript:
    """Tests for interview transcript chunking."""

    def test_chunk_interview_basic(self, sample_interview_text):
        """Test basic interview chunking."""
        chunks = chunk_interview_transcript(sample_interview_text, chunk_size=500, overlap=100)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_interview_speaker_preservation(self, sample_interview_text):
        """Test that speaker labels are preserved in chunks."""
        chunks = chunk_interview_transcript(sample_interview_text, chunk_size=2000, overlap=200)

        # At least one chunk should contain speaker labels
        has_speaker = any("INTERVIEWER" in c.text or "CANDIDATE" in c.text for c in chunks)
        assert has_speaker

    def test_chunk_interview_metadata(self, sample_interview_text):
        """Test that chunks have proper metadata."""
        chunks = chunk_interview_transcript(sample_interview_text, chunk_size=500, overlap=100)

        for chunk in chunks:
            assert "speakers" in chunk.metadata
            assert isinstance(chunk.metadata["speakers"], list)
            assert len(chunk.metadata["speakers"]) > 0

    def test_chunk_interview_timestamps(self, sample_interview_text):
        """Test that timestamps are preserved."""
        chunks = chunk_interview_transcript(sample_interview_text, chunk_size=2000, overlap=200)

        # At least one chunk should have a timestamp
        has_timestamp = any(c.timestamp for c in chunks)
        assert has_timestamp

    def test_chunk_interview_empty(self):
        """Test chunking empty transcript."""
        chunks = chunk_interview_transcript("", chunk_size=500, overlap=100)
        assert chunks == []


# ===========================================================================
# Sentence Splitting Tests
# ===========================================================================


class TestSplitIntoSentences:
    """Tests for sentence splitting."""

    def test_split_basic(self):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_split_empty(self):
        """Test splitting empty text."""
        assert split_into_sentences("") == []

    def test_split_no_terminators(self):
        """Test text without sentence terminators."""
        text = "This is just one long string"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1


# ===========================================================================
# Speaker Splitting Tests
# ===========================================================================


class TestSplitBySpeaker:
    """Tests for speaker-based splitting."""

    def test_split_by_speaker_basic(self, sample_interview_text):
        """Test basic speaker splitting."""
        blocks = split_by_speaker(sample_interview_text)

        assert len(blocks) > 0
        assert all("speaker" in b for b in blocks)
        assert all("text" in b for b in blocks)

    def test_split_by_speaker_labels(self, sample_interview_text):
        """Test that speaker labels are correctly identified."""
        blocks = split_by_speaker(sample_interview_text)

        speakers = {b["speaker"] for b in blocks}
        assert "INTERVIEWER" in speakers or "CANDIDATE" in speakers

    def test_split_by_speaker_no_labels(self):
        """Test splitting text without speaker labels."""
        text = "Just some text without any speaker labels at all."
        blocks = split_by_speaker(text)

        assert len(blocks) == 1
        assert blocks[0]["speaker"] == "UNKNOWN"

    def test_split_by_speaker_qa_format(self):
        """Test splitting Q&A format."""
        text = "Q: What is your experience?\nA: I have 5 years of experience."
        blocks = split_by_speaker(text)

        assert len(blocks) == 2
        assert blocks[0]["speaker"] == "Q"
        assert blocks[1]["speaker"] == "A"
