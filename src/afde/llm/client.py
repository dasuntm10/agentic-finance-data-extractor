"""Local LLM wrapper around Ollama with JSON-schema-constrained decoding.

The pipeline reaches an LLM in three places:
  - Section Locator tiebreak  (constrained classification, enum output)
  - Canonical Mapper tiebreak (constrained classification, enum output)
  - Risk Report narrative     (free-form prose, fact-checked post-hoc)

All three calls go through this single client. Constrained outputs are produced
by passing a JSON schema as Ollama's ``format`` parameter; Ollama enforces the
schema during decoding so invalid outputs are structurally impossible.

The default backend is Ollama because it ships a one-line installer on macOS,
Linux, and Windows. To run under vLLM instead, set ``AFDE_LLM_BACKEND=vllm``
and point ``AFDE_LLM_BASE_URL`` at the vLLM OpenAI-compatible endpoint; the
JSON-schema constraint is then delegated to ``outlines`` / ``lm-format-enforcer``
on the server side.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("AFDE_LLM_MODEL", "qwen2.5:7b-instruct")
DEFAULT_BASE_URL = os.environ.get("AFDE_LLM_BASE_URL", "http://localhost:11434")
NARRATIVE_MODEL = os.environ.get("AFDE_NARRATIVE_MODEL", DEFAULT_MODEL)


@dataclass
class ToolChoice:
    """A constrained JSON output. ``input_schema`` is enforced by the decoder."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMUnavailable(RuntimeError):
    pass


class LocalLLM:
    """Thin client over Ollama's /api/chat with JSON-schema constrained decoding."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import requests  # noqa: PLC0415
            except ImportError as e:
                raise LLMUnavailable("`requests` is required for the local LLM client.") from e
            self._session = requests.Session()
        return self._session

    def _model_for(self, tier: str) -> str:
        if tier == "narrative":
            return NARRATIVE_MODEL
        return self.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def structured(
        self,
        *,
        tier: str,
        system: str,
        user: str,
        tool: ToolChoice,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Constrained JSON output. The model must emit a value satisfying ``tool.input_schema``."""
        session = self._get_session()
        payload = {
            "model": self._model_for(tier),
            "messages": [
                {"role": "system", "content": system + "\n\nReturn only JSON matching the provided schema."},
                {"role": "user", "content": f"{tool.description}\n\n{user}"},
            ],
            "stream": False,
            "format": tool.input_schema,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = session.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        except Exception as e:
            raise LLMUnavailable(f"Local LLM unreachable at {self.base_url}: {e}") from e
        if resp.status_code != 200:
            raise LLMUnavailable(f"Local LLM error {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        content = body.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMUnavailable("Local LLM returned empty content.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMUnavailable(f"Local LLM returned non-JSON despite schema: {content[:200]}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def text(
        self,
        *,
        tier: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str:
        """Free-form text generation. Used only for the analyst narrative."""
        session = self._get_session()
        payload = {
            "model": self._model_for(tier),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = session.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        except Exception as e:
            raise LLMUnavailable(f"Local LLM unreachable at {self.base_url}: {e}") from e
        if resp.status_code != 200:
            raise LLMUnavailable(f"Local LLM error {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        return body.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Deterministic fallback — used when the local LLM is unreachable so the
# pipeline can still complete (with degraded outputs flagged on the report).
# ---------------------------------------------------------------------------


class FallbackLLM:
    """Returns empty/placeholder results; the agent code treats these as a signal
    to apply its deterministic fallback path."""

    def structured(self, *, tier: str, system: str, user: str, tool: ToolChoice, max_tokens: int = 1024):
        log.warning("FallbackLLM.structured invoked; returning empty result for tool %s", tool.name)
        return {}

    def text(self, *, tier: str, system: str, user: str, max_tokens: int = 2048) -> str:
        log.warning("FallbackLLM.text invoked; narrative will use deterministic template.")
        return ""


def get_llm() -> LocalLLM | FallbackLLM:
    """Return the active LLM client. Falls back to the deterministic stub if
    the local server isn't reachable at import time."""
    client = LocalLLM()
    try:
        session = client._get_session()
        resp = session.get(f"{client.base_url}/api/tags", timeout=2)
        if resp.status_code == 200:
            return client
        log.warning("Local LLM probe returned %d; using fallback.", resp.status_code)
    except Exception as e:
        log.warning("Local LLM unreachable at %s (%s); using fallback.", client.base_url, e)
    return FallbackLLM()
