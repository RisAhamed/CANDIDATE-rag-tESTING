"""Singleton Groq LLM client wrapper using the OpenAI-compatible endpoint."""

import json
import re
from typing import Any, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.logging import logger


class GroqClient:
    """Singleton Groq client for LLM interactions.

    Uses the OpenAI-compatible endpoint exposed by Groq. The API key is
    loaded from settings and is never logged.
    """

    _instance: Optional["GroqClient"] = None
    _client: Optional[OpenAI] = None

    def __new__(cls) -> "GroqClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._client is None:
            if not settings.GROQ_API_KEY:
                logger.warning(
                    "GROQ_API_KEY is not set. LLM features will be unavailable. "
                    "Set GROQ_API_KEY in your .env file."
                )
                return
            logger.info(
                f"Initializing Groq client (base_url={settings.GROQ_BASE_URL}, "
                f"model={settings.GROQ_MODEL})"
            )
            self._client = OpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url=settings.GROQ_BASE_URL,
            )
            logger.info("Groq client initialized")

    @property
    def client(self) -> OpenAI:
        """Get the underlying OpenAI client."""
        if self._client is None:
            raise RuntimeError("Groq client has not been initialized")
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, str]] = None,
    ) -> str:
        """Call the OpenAI chat completions API via Groq.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model override. Defaults to settings.GROQ_MODEL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional response format (e.g. {"type": "json_object"}).

        Returns:
            The generated text content.

        Raises:
            RuntimeError: If the API call fails after retries.
        """
        model = model or settings.GROQ_MODEL

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        logger.debug(
            f"Groq chat_completion request: model={model}, "
            f"messages={len(messages)}, temperature={temperature}, "
            f"max_tokens={max_tokens}, response_format={response_format}"
        )

        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug(
                f"Groq chat_completion response: {len(content)} chars, "
                f"finish_reason={response.choices[0].finish_reason}"
            )
            return content
        except Exception as e:
            logger.error(f"Groq chat_completion failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"Groq API call failed: {e}") from e

    @staticmethod
    def parse_json_response(response_text: str) -> dict[str, Any]:
        """Extract a JSON object from an LLM response.

        Handles raw JSON, markdown code blocks (```json ... ```), and
        surrounding prose.

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            Parsed JSON as a dict.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
        if not response_text or not response_text.strip():
            raise ValueError("Empty response text")

        text = response_text.strip()

        # 1) Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2) Try to extract from markdown code blocks
        code_block_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            re.DOTALL | re.IGNORECASE,
        )
        match = code_block_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3) Try to find the first balanced JSON object in the text
        brace_pattern = re.compile(r"\{", re.DOTALL)
        for match in brace_pattern.finditer(text):
            candidate = text[match.start():]
            # Try progressively shorter substrings
            for end in range(len(candidate), 0, -1):
                try:
                    return json.loads(candidate[:end])
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not extract valid JSON from response: {response_text[:200]}..."
        )


# Global singleton instance
groq_client = GroqClient()
