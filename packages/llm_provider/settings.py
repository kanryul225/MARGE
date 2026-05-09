"""Environment-driven settings for the LLM provider abstraction.

The orchestrator and the medical-expert agent both read settings from here
so a single env var change swaps the entire LLM stack.

Env vars:
- LLM_PROVIDER         — "anthropic" (default) or "watsonx"
- ANTHROPIC_API_KEY    — required when provider=anthropic
- ANTHROPIC_MODEL_ID   — optional, defaults to a Claude Haiku snapshot
- WATSONX_API_KEY      — required when provider=watsonx
- WATSONX_PROJECT_ID   — required when provider=watsonx
- WATSONX_URL          — required when provider=watsonx
- WATSONX_MODEL_ID     — optional, defaults to a Granite instruct model
"""

import os
from dataclasses import dataclass
from enum import Enum

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_WATSONX_MODEL = "ibm/granite-3-8b-instruct"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    WATSONX = "watsonx"


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration."""

    provider: Provider
    model_id: str
    api_key: str | None = None
    base_url: str | None = None
    project_id: str | None = None  # watsonx-only

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider_str = os.getenv("LLM_PROVIDER", Provider.ANTHROPIC.value).lower()
        try:
            provider = Provider(provider_str)
        except ValueError as e:
            valid = [p.value for p in Provider]
            raise ValueError(
                f"Unknown LLM provider: {provider_str!r}. Expected one of {valid}."
            ) from e

        if provider is Provider.ANTHROPIC:
            return cls(
                provider=provider,
                model_id=os.getenv("ANTHROPIC_MODEL_ID", _DEFAULT_ANTHROPIC_MODEL),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        # provider is Provider.WATSONX
        return cls(
            provider=provider,
            model_id=os.getenv("WATSONX_MODEL_ID", _DEFAULT_WATSONX_MODEL),
            api_key=os.getenv("WATSONX_API_KEY"),
            base_url=os.getenv("WATSONX_URL"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
        )
