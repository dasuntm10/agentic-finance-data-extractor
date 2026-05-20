"""Claude wrapper with tiered model selection and forced tool_use for structured outputs.

Tiers:
  - "haiku"  → classification tiebreaks (Section Locator, Canonical Mapper). Strict enum outputs.
  - "sonnet" → analyst narrative. Free-form prose constrained by fact-check post-pass.
  - "opus"   → escalation on a second narrative hallucination-check failure. Not a default.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from afde.config import anthropic_api_key

log = logging.getLogger(__name__)

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


@dataclass
class ToolChoice:
    """A forced tool call with a schema. The model MUST return JSON matching `input_schema`."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMUnavailable(RuntimeError):
    pass


class ClaudeClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or anthropic_api_key()
        self._client = None

    def _get(self):
        if self._client is None:
            if not self.api_key:
                raise LLMUnavailable("ANTHROPIC_API_KEY not set. Export it or run with --offline.")
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise LLMUnavailable("anthropic SDK not installed; `poetry install`.") from e
            self._client = Anthropic(api_key=self.api_key)
        return self._client

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
        """Force a single tool_use call. Returns the validated tool input dict."""
        model = MODELS[tier]
        client = self._get()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}],
            tool_choice={"type": "tool", "name": tool.name},
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool.name:
                return dict(block.input)
        raise LLMUnavailable(f"Model {model} returned no tool_use block: {resp}")

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
        model = MODELS[tier]
        client = self._get()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Offline fallback (stub) — wired so agent code does not branch on profile.
# ---------------------------------------------------------------------------


class OfflineLLM:
    """Returns deterministic defaults; used when AFDE_OFFLINE=1 or no API key.

    The real --offline profile would route to a local Qwen/Llama via vLLM/Ollama;
    this stub keeps the pipeline runnable on a developer machine without keys.
    """

    def structured(self, *, tier: str, system: str, user: str, tool: ToolChoice, max_tokens: int = 1024):
        log.warning("OfflineLLM.structured invoked; returning empty result for tool %s", tool.name)
        return {}

    def text(self, *, tier: str, system: str, user: str, max_tokens: int = 2048) -> str:
        log.warning("OfflineLLM.text invoked; returning placeholder narrative.")
        return "(Narrative generation skipped — running in offline mode without local LLM configured.)"


def get_llm() -> ClaudeClient | OfflineLLM:
    from afde.config import is_offline

    if is_offline() or not anthropic_api_key():
        return OfflineLLM()
    return ClaudeClient()
