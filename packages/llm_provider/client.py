"""Factory for BeeAI ChatModel instances.

`build_chat_model(settings)` returns a configured ChatModel for the chosen
provider. Both `apps/orchestrator/agent.py` and
`services/medical_expert_agent/agent.py` (next slice) consume this — they
never import provider SDKs directly.

This is the only place where provider-specific code lives. Adding a new
provider = a new `if` branch here, plus the provider name in
`packages/llm_provider/settings.Provider`.
"""

from typing import TYPE_CHECKING

from packages.llm_provider.settings import LLMSettings, Provider

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel


def build_chat_model(settings: LLMSettings) -> "ChatModel":
    if settings.provider is Provider.ANTHROPIC:
        return _build_anthropic(settings)
    if settings.provider is Provider.WATSONX:
        return _build_watsonx(settings)
    raise AssertionError(f"Unhandled provider: {settings.provider!r}")


def _build_anthropic(settings: LLMSettings) -> "ChatModel":
    if not settings.api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
        )
    from beeai_framework.adapters.anthropic.backend.chat import AnthropicChatModel

    return AnthropicChatModel(
        model_id=settings.model_id,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def _build_watsonx(settings: LLMSettings) -> "ChatModel":
    if not settings.api_key:
        raise ValueError("WATSONX_API_KEY is required when LLM_PROVIDER=watsonx.")
    if not settings.project_id:
        raise ValueError("WATSONX_PROJECT_ID is required when LLM_PROVIDER=watsonx.")
    if not settings.base_url:
        raise ValueError("WATSONX_URL is required when LLM_PROVIDER=watsonx.")
    from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel

    return WatsonxChatModel(
        model_id=settings.model_id,
        api_key=settings.api_key,
        project_id=settings.project_id,
        base_url=settings.base_url,
    )
