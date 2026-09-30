"""Text parsing and normalization utilities."""

import re

from app.core.logging import logger


def normalize_text(text: str) -> str:
    """
    Normalize extracted text:
    - Remove excessive whitespace
    - Remove obvious garbage characters
    - Preserve meaningful headings
    - Normalize line endings
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and other control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces on lines
    text = re.sub(r" {2,}", " ", text)

    # Remove lines that are just punctuation or symbols
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r"^[\W_]+$", stripped):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_name(text: str) -> str:
    """Extract candidate name from text."""
    lines = text.strip().split("\n")

    # Look for "Name:" pattern
    for line in lines[:10]:
        match = re.match(r"(?:name|full\s*name)\s*:\s*(.+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Look for "Resume of X" or "CV of X"
    for line in lines[:5]:
        match = re.match(r"(?:resume|cv|curriculum\s*vitae)\s*(?:of|for)\s*:?\s*(.+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Use first non-empty line if it looks like a name
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("resume", "cv", "curriculum vitae", "profile", "portfolio"):
            continue
        words = line.split()
        if 1 <= len(words) <= 5:
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.8:
                return line
        break

    return ""


def extract_email(text: str) -> str:
    """Extract first email from text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Extract first phone number from text."""
    patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}",
        r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def extract_experience_years(text: str) -> float:
    """Extract years of experience from text."""
    text_lower = text.lower()
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s+)?experience",
        r"experience\s*:?\s*(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yrs?\s*(?:of\s+)?exp",
        r"work(?:ed|ing)?\s*for\s*(\d+)\+?\s*years?",
        r"over\s*(\d+)\+?\s*years?",
        r"more\s*than\s*(\d+)\+?\s*years?",
    ]

    years_list = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                years = float(match)
                if 0 < years <= 50:
                    years_list.append(years)
            except ValueError:
                continue

    return max(years_list) if years_list else 0.0


def extract_skills(text: str) -> list[str]:
    """Extract skills from text using known skills list."""
    from app.ingestion.chunker import KNOWN_SKILLS

    text_lower = text.lower()
    found_skills = set()

    for skill in KNOWN_SKILLS:
        if len(skill) <= 4:
            pattern = r"\b" + re.escape(skill) + r"\b"
        else:
            pattern = re.escape(skill)
        if re.search(pattern, text_lower):
            found_skills.add(skill)

    return sorted(found_skills)


def extract_education(text: str) -> str:
    """Extract education information from text."""
    text_lower = text.lower()
    education_parts = []

    edu_keywords = [
        "bachelor", "master", "ph.d", "phd", "b.tech", "m.tech", "b.e.", "m.e.",
        "b.sc", "m.sc", "b.com", "m.com", "bba", "mba", "bca", "mca",
        "bachelor of", "master of", "doctor of", "associate", "diploma",
        "university", "institute", "college",
    ]

    for keyword in edu_keywords:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            start = max(0, idx - 30)
            end = min(len(text), idx + 80)
            context = text[start:end].strip()
            context = re.sub(r"\s+", " ", context)
            if context and context not in education_parts:
                education_parts.append(context)

    # Look for "Education" section
    edu_section = re.search(
        r"education\s*:?\s*(.+?)(?:\n\n|\n[A-Z]|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if edu_section:
        edu_text = edu_section.group(1).strip()
        edu_text = re.sub(r"\s+", " ", edu_text)
        if edu_text and len(edu_text) > 5:
            education_parts.insert(0, edu_text)

    return " | ".join(education_parts[:3])


def extract_companies(text: str) -> list[str]:
    """Extract company names from text."""
    companies = []
    patterns = [
        r"(?:at|@)\s+([A-Z][A-Za-z\s&.,]+?)(?:\s*\(|,|\n|$)",
        r"([A-Z][A-Za-z\s&.,]+?)\s*\(\d{4}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            company = match.strip()
            if len(company) > 2 and company not in companies:
                companies.append(company)
    return companies[:10]


def parse_candidate_profile(text: str, filename: str = "") -> dict:
    """
    Parse a candidate profile from resume text.

    Returns a dictionary with extracted fields.
    """
    profile = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "total_experience": extract_experience_years(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "companies": extract_companies(text),
        "resume_text": text,
        "profile_source": filename,
    }

    logger.info(
        f"Parsed profile: name={profile['name']}, "
        f"experience={profile['total_experience']}, "
        f"skills={len(profile['skills'])}"
    )
    return profile
