"""Natural language query parser using Groq LLM."""

import re
from typing import Optional

from app.core.logging import logger
from app.llm.groq_client import groq_client
from app.schemas.search import InterpretedRequirements

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise job-requirement parser for a candidate search system.
Convert the user's natural language query into a structured JSON object.

Extract the following fields:
- job_titles: list of job titles or roles mentioned (e.g. ["backend developer", "ML engineer"])
- required_skills: list of hard skills explicitly required (e.g. ["Python", "FastAPI", "PostgreSQL"])
- experience_years: a single number if the user says "N years of experience" (float or null)
- experience_min: minimum years of experience (float or null)
- experience_max: maximum years of experience (float or null)
- location: list of locations mentioned (e.g. ["Berlin", "Remote"])
- interview_required_skills: skills the user wants to verify from interview transcripts
- employment_type: e.g. "full-time", "contract", "freelance" (or null)
- semantic_query: a cleaned, concise version of the query optimized for vector/semantic search.
  Remove filler words, keep key terms, skills, and intent. This will be embedded.
- hard_constraints: requirements that are non-negotiable (e.g. "must have Kubernetes", "5+ years")
- soft_preferences: nice-to-have preferences (e.g. "experience with Rust is a plus")

Rules:
- Only extract what is explicitly stated or strongly implied. Do not invent requirements.
- If a field has no value, use an empty list [] or null.
- Return ONLY valid JSON with no markdown formatting.
"""


def _build_messages(query: str) -> list[dict[str, str]]:
    """Build the chat messages for the query parser."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


# ---------------------------------------------------------------------------
# Fallback keyword extraction
# ---------------------------------------------------------------------------

# Common tech keywords for fallback extraction
_COMMON_SKILLS = {
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "react", "angular", "vue", "svelte", "next.js", "nuxt", "django",
    "flask", "fastapi", "express", "spring", "spring boot", "rails",
    "node.js", "nodejs", "deno", "bun",
    "aws", "azure", "gcp", "google cloud", "heroku", "vercel",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "ci/cd", "github actions", "gitlab ci",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "sqlite", "dynamodb", "firebase", "supabase",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "data science", "data engineering", "data analysis", "analytics",
    "rest", "restful", "graphql", "grpc", "websocket",
    "microservices", "serverless", "lambda", "event-driven",
    "agile", "scrum", "kanban", "jira",
    "linux", "unix", "bash", "shell scripting",
    "html", "css", "sass", "tailwind", "bootstrap",
    "figma", "sketch", "photoshop", "ui/ux", "user research",
    "tableau", "power bi", "looker", "excel",
    "hadoop", "spark", "kafka", "airflow", "dbt",
    "security", "penetration testing", "oauth", "jwt", "encryption",
}

# Common job title keywords
_COMMON_TITLES = {
    "software engineer", "software developer", "backend developer",
    "backend engineer", "frontend developer", "frontend engineer",
    "full stack developer", "fullstack developer", "full-stack developer",
    "devops engineer", "sre", "site reliability engineer",
    "data scientist", "data engineer", "data analyst",
    "machine learning engineer", "ml engineer", "ai engineer",
    "product manager", "project manager", "engineering manager",
    "tech lead", "team lead", "architect", "cto", "vp of engineering",
    "qa engineer", "test engineer", "automation engineer",
    "mobile developer", "ios developer", "android developer",
    "security engineer", "cloud engineer", "platform engineer",
    "database administrator", "dba", "sysadmin",
    "designer", "ux designer", "ui designer", "product designer",
    "recruiter", "hr", "human resources",
    "intern", "junior", "senior", "staff", "principal",
}

# Common location keywords
_COMMON_LOCATIONS = {
    "remote", "hybrid", "onsite", "on-site",
    "berlin", "london", "paris", "amsterdam", "munich", "hamburg",
    "new york", "san francisco", "seattle", "austin", "boston",
    "toronto", "vancouver", "montreal",
    "bangalore", "mumbai", "delhi", "hyderabad", "singapore",
    "tokyo", "sydney", "melbourne",
    "germany", "uk", "usa", "us", "canada", "india", "australia",
    "netherlands", "france", "spain", "italy", "switzerland",
}

# Employment type keywords
_EMPLOYMENT_TYPES = {
    "full-time": "full-time",
    "full time": "full-time",
    "part-time": "part-time",
    "part time": "part-time",
    "contract": "contract",
    "contractor": "contract",
    "freelance": "freelance",
    "freelancer": "freelance",
    "internship": "internship",
    "intern": "internship",
    "temporary": "temporary",
}


def _fallback_parse(query: str) -> InterpretedRequirements:
    """Basic keyword-based fallback when the LLM is unavailable.

    This is intentionally simple — it only catches obvious patterns so the
    search pipeline can still function without an LLM.
    """
    query_lower = query.lower()
    words = re.findall(r"[a-zA-Z+#]+", query_lower)

    # Extract skills (filter out single-character skills to avoid false positives)
    found_skills: list[str] = []
    for skill in _COMMON_SKILLS:
        if len(skill) >= 2 and skill in query_lower:
            found_skills.append(skill)

    # Extract job titles
    found_titles: list[str] = []
    for title in _COMMON_TITLES:
        if title in query_lower:
            found_titles.append(title)

    # Extract locations
    found_locations: list[str] = []
    for loc in _COMMON_LOCATIONS:
        if loc in query_lower:
            found_locations.append(loc)

    # Extract employment type
    emp_type: Optional[str] = None
    for keyword, normalized in _EMPLOYMENT_TYPES.items():
        if keyword in query_lower:
            emp_type = normalized
            break

    # Extract experience years
    exp_years: Optional[float] = None
    exp_min: Optional[float] = None
    exp_max: Optional[float] = None

    # Pattern: "N years" or "N+ years"
    exp_match = re.search(r"(\d+)\+?\s*years?", query_lower)
    if exp_match:
        exp_years = float(exp_match.group(1))

    # Pattern: "N to M years" or "N-M years"
    range_match = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*years?", query_lower)
    if range_match:
        exp_min = float(range_match.group(1))
        exp_max = float(range_match.group(2))

    # Pattern: "at least N years"
    min_match = re.search(r"at least\s*(\d+)\s*years?", query_lower)
    if min_match:
        exp_min = float(min_match.group(1))

    # Pattern: "at most N years" or "up to N years"
    max_match = re.search(r"(?:at most|up to)\s*(\d+)\s*years?", query_lower)
    if max_match:
        exp_max = float(max_match.group(1))

    # Build a cleaned semantic query: remove filler words
    filler_words = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "the", "a", "an", "and", "or", "but", "if", "then",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "shall", "to", "of", "in", "on", "at",
        "for", "with", "by", "from", "as", "about", "into", "through",
        "looking", "search", "find", "need", "want", "please", "someone",
        "candidate", "person", "developer", "engineer", "who", "that",
        "this", "these", "those", "there", "here", "what", "which",
        "when", "where", "how", "why", "not", "no", "yes", "all", "any",
        "some", "each", "every", "both", "few", "more", "most", "other",
        "such", "only", "own", "same", "so", "than", "too", "very",
        "just", "also", "now", "even", "still", "already", "yet",
    }
    semantic_words = [w for w in words if w not in filler_words]
    semantic_query = " ".join(semantic_words) if semantic_words else query

    return InterpretedRequirements(
        job_titles=found_titles,
        required_skills=found_skills,
        experience_years=exp_years,
        experience_min=exp_min,
        experience_max=exp_max,
        location=found_locations,
        interview_required_skills=[],
        employment_type=emp_type,
        semantic_query=semantic_query,
        hard_constraints=[],
        soft_preferences=[],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_query(query: str) -> InterpretedRequirements:
    """Parse a natural language query into structured requirements.

    Uses the Groq LLM to extract structured fields. Falls back to basic
    keyword extraction if the LLM call fails.

    Args:
        query: Natural language search query.

    Returns:
        InterpretedRequirements with extracted fields.
    """
    if not query or not query.strip():
        logger.warning("Empty query received, returning empty requirements")
        return InterpretedRequirements(semantic_query=query)

    messages = _build_messages(query)

    try:
        response_text = groq_client.chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        parsed = groq_client.parse_json_response(response_text)

        # Validate and construct the schema
        requirements = InterpretedRequirements(
            job_titles=parsed.get("job_titles", []),
            required_skills=parsed.get("required_skills", []),
            experience_years=parsed.get("experience_years"),
            experience_min=parsed.get("experience_min"),
            experience_max=parsed.get("experience_max"),
            location=parsed.get("location", []),
            interview_required_skills=parsed.get("interview_required_skills", []),
            employment_type=parsed.get("employment_type"),
            semantic_query=parsed.get("semantic_query", query),
            hard_constraints=parsed.get("hard_constraints", []),
            soft_preferences=parsed.get("soft_preferences", []),
        )

        logger.info(
            f"Query parsed: titles={requirements.job_titles}, "
            f"skills={requirements.required_skills}, "
            f"exp={requirements.experience_min}-{requirements.experience_max}, "
            f"location={requirements.location}, "
            f"semantic_query='{requirements.semantic_query[:80]}...'"
        )
        return requirements

    except Exception as e:
        logger.warning(
            f"LLM query parsing failed ({type(e).__name__}: {e}). "
            f"Falling back to keyword extraction."
        )
        fallback = _fallback_parse(query)
        logger.info(
            f"Fallback parse: titles={fallback.job_titles}, "
            f"skills={fallback.required_skills}, "
            f"exp={fallback.experience_min}-{fallback.experience_max}"
        )
        return fallback
