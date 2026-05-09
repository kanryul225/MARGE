"""Tests for the LLM provider abstraction.

`LLMSettings` reads environment variables and exposes a typed config.
`build_chat_model(settings)` returns a BeeAI ChatModel for the configured
provider — defaulting to Anthropic, with watsonx as the swappable target.

Tests verify settings parsing and provider selection. They do not actually
issue an LLM call (no live network required).
"""

import os
from unittest.mock import patch

import pytest

from packages.llm_provider.client import build_chat_model
from packages.llm_provider.settings import LLMSettings, Provider


class TestLLMSettings:
    def test_defaults_to_anthropic(self):
        with patch.dict(os.environ, {}, clear=True):
            s = LLMSettings.from_env()
            assert s.provider == Provider.ANTHROPIC

    def test_reads_provider_from_env(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "watsonx"}, clear=True):
            s = LLMSettings.from_env()
            assert s.provider == Provider.WATSONX

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "no_such_one"}, clear=True):
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                LLMSettings.from_env()

    def test_anthropic_model_default(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            s = LLMSettings.from_env()
            assert s.model_id  # default model id is set
            assert s.model_id.startswith("claude-")

    def test_anthropic_api_key_read(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-test"},
            clear=True,
        ):
            s = LLMSettings.from_env()
            assert s.api_key == "sk-test"

    def test_watsonx_settings_read(self):
        env = {
            "LLM_PROVIDER": "watsonx",
            "WATSONX_API_KEY": "wx-test",
            "WATSONX_PROJECT_ID": "proj-1",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_MODEL_ID": "ibm/granite-3-8b-instruct",
        }
        with patch.dict(os.environ, env, clear=True):
            s = LLMSettings.from_env()
            assert s.provider == Provider.WATSONX
            assert s.api_key == "wx-test"
            assert s.project_id == "proj-1"
            assert s.base_url == "https://us-south.ml.cloud.ibm.com"
            assert s.model_id == "ibm/granite-3-8b-instruct"


class TestBuildChatModel:
    def test_returns_anthropic_chat_model(self):
        s = LLMSettings(
            provider=Provider.ANTHROPIC,
            model_id="claude-haiku-4-5-20251001",
            api_key="sk-test",
        )
        model = build_chat_model(s)
        # Don't actually issue any LLM calls — just verify the right adapter was chosen.
        from beeai_framework.adapters.anthropic.backend.chat import AnthropicChatModel

        assert isinstance(model, AnthropicChatModel)

    def test_returns_watsonx_chat_model(self):
        s = LLMSettings(
            provider=Provider.WATSONX,
            model_id="ibm/granite-3-8b-instruct",
            api_key="wx-test",
            project_id="proj-1",
            base_url="https://us-south.ml.cloud.ibm.com",
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel

        assert isinstance(model, WatsonxChatModel)

    def test_anthropic_requires_api_key(self):
        s = LLMSettings(provider=Provider.ANTHROPIC, model_id="claude-x")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            build_chat_model(s)

    def test_watsonx_requires_project_id(self):
        s = LLMSettings(
            provider=Provider.WATSONX, model_id="x", api_key="k", base_url="u"
        )
        with pytest.raises(ValueError, match="WATSONX_PROJECT_ID"):
            build_chat_model(s)
