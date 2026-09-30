"""Candidate summary generator using Groq LLM."""

from typing import Optional

from app.core.logging import logger
from app.llm.groq_client import groq_client
from app.schemas.search import InterpretedRequirements

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """You are a technical recruiter assistant. Generate a concise, factual summary of a candidate based on the provided profile data, interview evidence, and job requirements.

Your summary MUST:
- Explain why the candidate matches the requirements
- State which requirements are supported by evidence
- Clearly indicate which evidence comes from the interview vs. the profile
- Note what is missing or not covered
- State what remains uncertain

Your summary MUST NOT:
- Invent or assume any information not present in the provided context
- Exaggerate the candidate's qualifications
- Make claims without supporting evidence

Keep the summary concise: 2-4 sentences. Use clear, professional language.
If evidence is insufficient to make a judgment, explicitly state that.
"""

_COMPARISON_SYSTEM_PROMPT = """You are a technical recruiter assistant. Compare the top candidates based on the provided profiles, interview evidence, and job requirements.

Your comparison MUST:
- Rank the candidates from best to worst match
- Explain the key differentiators between candidates
- Note which requirements each candidate meets or lacks
- Highlight any concerns or red flags
- Be fair and objective — only use the provided evidence

Your comparison MUST NOT:
- Invent or assume any information not present in the provided context
- Show bias toward any candidate

Keep the comparison concise: 3-5 sentences per candidate, plus a brief overall recommendation.
"""


def _format_candidate_context(
    candidate_data: dict,
    requirements: InterpretedRequirements,
) -> str:
    """Format candidate data and requirements into a text block for the LLM."""
    lines: list[str] = []

    # Requirements
    lines.append("=== JOB REQUIREMENTS ===")
    if requirements.job_titles:
        lines.append(f"Job titles: {', '.join(requirements.job_titles)}")
    if requirements.required_skills:
        lines.append(f"Required skills: {', '.join(requirements.required_skills)}")
    if requirements.experience_min is not None or requirements.experience_max is not None:
        exp_range = f"{requirements.experience_min or 0}-{requirements.experience_max or 'any'} years"
        lines.append(f"Experience: {exp_range}")
    if requirements.location:
        lines.append(f"Location: {', '.join(requirements.location)}")
    if requirements.employment_type:
        lines.append(f"Employment type: {requirements.employment_type}")
    if requirements.hard_constraints:
        lines.append(f"Hard constraints: {', '.join(requirements.hard_constraints)}")
    if requirements.soft_preferences:
        lines.append(f"Soft preferences: {', '.join(requirements.soft_preferences)}")
    lines.append("")

    # Candidate profile
    lines.append("=== CANDIDATE PROFILE ===")
    for key, value in candidate_data.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if sub_value:
                    lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")

    return "\n".join(lines)


def _format_comparison_context(
    candidates: list[dict],
    requirements: InterpretedRequirements,
) -> str:
    """Format multiple candidates and requirements for comparison."""
    lines: list[str] = []

    # Requirements
    lines.append("=== JOB REQUIREMENTS ===")
    if requirements.job_titles:
        lines.append(f"Job titles: {', '.join(requirements.job_titles)}")
    if requirements.required_skills:
        lines.append(f"Required skills: {', '.join(requirements.required_skills)}")
    if requirements.experience_min is not None or requirements.experience_max is not None:
        exp_range = f"{requirements.experience_min or 0}-{requirements.experience_max or 'any'} years"
        lines.append(f"Experience: {exp_range}")
    if requirements.location:
        lines.append(f"Location: {', '.join(requirements.location)}")
    if requirements.employment_type:
        lines.append(f"Employment type: {requirements.employment_type}")
    if requirements.hard_constraints:
        lines.append(f"Hard constraints: {', '.join(requirements.hard_constraints)}")
    if requirements.soft_preferences:
        lines.append(f"Soft preferences: {', '.join(requirements.soft_preferences)}")
    lines.append("")

    # Each candidate
    for i, candidate in enumerate(candidates, 1):
        lines.append(f"=== CANDIDATE {i} ===")
        for key, value in candidate.items():
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    if sub_value:
                        lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback template-based summary
# ---------------------------------------------------------------------------

def _fallback_summary(
    candidate_data: dict,
    requirements: InterpretedRequirements,
) -> str:
    """Generate a basic template-based summary when the LLM is unavailable."""
    name = candidate_data.get("name", "Candidate")
    title = candidate_data.get("title", "")
    experience = candidate_data.get("experience_years", 0)
    skills = candidate_data.get("skills", [])
    location = candidate_data.get("location", "")

    parts: list[str] = []

    # Title and experience
    if title and experience:
        parts.append(f"{name} is a {title} with {experience} years of experience.")
    elif title:
        parts.append(f"{name} is a {title}.")
    else:
        parts.append(f"{name} has {experience} years of experience." if experience else f"{name} is a candidate.")

    # Skills match
    if skills and requirements.required_skills:
        candidate_skills_lower = {s.lower() for s in skills}
        matched = [s for s in requirements.required_skills if s.lower() in candidate_skills_lower]
        missing = [s for s in requirements.required_skills if s.lower() not in candidate_skills_lower]
        if matched:
            parts.append(f"Matched skills: {', '.join(matched)}.")
        if missing:
            parts.append(f"Missing skills: {', '.join(missing)}.")
    elif skills:
        parts.append(f"Skills: {', '.join(skills[:10])}.")

    # Location
    if location and requirements.location:
        loc_match = any(loc.lower() in location.lower() for loc in requirements.location)
        if loc_match:
            parts.append(f"Location matches: {location}.")
        else:
            parts.append(f"Location: {location} (may not match requirements).")

    # Experience check
    if experience and requirements.experience_min is not None:
        if experience >= requirements.experience_min:
            parts.append(f"Experience meets minimum requirement ({requirements.experience_min}+ years).")
        else:
            parts.append(f"Experience below minimum requirement ({requirements.experience_min}+ years).")

    if not parts:
        return f"{name} — insufficient data to generate a meaningful summary."

    return " ".join(parts)


def _fallback_comparison_summary(
    candidates: list[dict],
    requirements: InterpretedRequirements,
) -> str:
    """Generate a basic template-based comparison when the LLM is unavailable."""
    if not candidates:
        return "No candidates to compare."

    lines: list[str] = ["Candidate Comparison (fallback):", ""]

    for i, candidate in enumerate(candidates, 1):
        name = candidate.get("name", f"Candidate {i}")
        title = candidate.get("title", "Unknown role")
        experience = candidate.get("experience_years", 0)
        skills = candidate.get("skills", [])

        # Count matched skills
        matched_count = 0
        if skills and requirements.required_skills:
            candidate_skills_lower = {s.lower() for s in skills}
            matched_count = sum(
                1 for s in requirements.required_skills if s.lower() in candidate_skills_lower
            )

        lines.append(
            f"{i}. {name} — {title}, {experience} years exp, "
            f"{matched_count}/{len(requirements.required_skills)} required skills matched."
        )

    lines.append("")
    lines.append("Note: This is a fallback summary. LLM-powered comparison is unavailable.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def generate_candidate_summary(
    candidate_data: dict,
    requirements: InterpretedRequirements,
) -> str:
    """Generate a concise recruiter-facing summary for a single candidate.

    Uses the Groq LLM to produce a summary explaining why the candidate
    matches (or doesn't match) the requirements. Falls back to a
    template-based summary if the LLM call fails.

    Args:
        candidate_data: Dictionary containing candidate profile data and
            interview evidence.
        requirements: The interpreted job requirements.

    Returns:
        A concise summary string (2-4 sentences).
    """
    if not candidate_data:
        logger.warning("Empty candidate data provided for summary generation")
        return "No candidate data available."

    context = _format_candidate_context(candidate_data, requirements)
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    try:
        summary = groq_client.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        summary = summary.strip()
        logger.info(
            f"Generated candidate summary for "
            f"'{candidate_data.get('name', 'unknown')}': {len(summary)} chars"
        )
        return summary

    except Exception as e:
        logger.warning(
            f"LLM summary generation failed ({type(e).__name__}: {e}). "
            f"Falling back to template-based summary."
        )
        return _fallback_summary(candidate_data, requirements)


def generate_comparison_summary(
    candidates: list[dict],
    requirements: InterpretedRequirements,
) -> str:
    """Generate a comparison summary for multiple top candidates.

    Uses the Groq LLM to compare candidates and provide a ranking with
    key differentiators. Falls back to a template-based comparison if
    the LLM call fails.

    Args:
        candidates: List of candidate data dictionaries (typically top 3-5).
        requirements: The interpreted job requirements.

    Returns:
        A comparison summary string.
    """
    if not candidates:
        logger.warning("No candidates provided for comparison")
        return "No candidates to compare."

    if len(candidates) == 1:
        return generate_candidate_summary(candidates[0], requirements)

    context = _format_comparison_context(candidates, requirements)
    messages = [
        {"role": "system", "content": _COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    try:
        summary = groq_client.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        summary = summary.strip()
        logger.info(
            f"Generated comparison summary for {len(candidates)} candidates: "
            f"{len(summary)} chars"
        )
        return summary

    except Exception as e:
        logger.warning(
            f"LLM comparison generation failed ({type(e).__name__}: {e}). "
            f"Falling back to template-based comparison."
        )
        return _fallback_comparison_summary(candidates, requirements)
