"""Tests for medical expert implementations."""

import pytest

from packages.schemas.retrieval import Citation, MedicalExpertResponse
from services.medical_expert_agent.agent import (
    MedicalExpertAgent,
    StubMedicalExpert,
    build_medical_expert_agent,
)


class _FakeResult:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text_content(self) -> str:
        return self._text


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages = None

    async def run(self, messages):
        self.messages = messages
        return _FakeResult(self.text)


class TestStubMedicalExpert:
    def test_returns_medical_expert_response(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="any question", findings={})
        assert isinstance(response, MedicalExpertResponse)

    def test_response_has_non_empty_reasoning(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.reasoning
        assert len(response.reasoning) > 20

    def test_response_has_at_least_one_citation(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert len(response.citations) >= 1
        assert isinstance(response.citations[0], Citation)

    def test_citation_has_source_url(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.citations[0].document.source_url

    def test_not_abstained_by_default(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.abstained is False

    def test_response_independent_of_input(self):
        expert = StubMedicalExpert()
        a = expert.consult(question="q1", findings={"x": 1})
        b = expert.consult(question="q2", findings={"y": 2})
        assert a.reasoning == b.reasoning


class TestMedicalExpertAgent:
    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_medical_expert_response(self):
        llm = _FakeLLM(
            '{"reasoning":"The supplied glucose and BMI warrant follow-up.",'
            '"abstained":false,"abstain_reason":null,"citations":[]}'
        )
        expert = MedicalExpertAgent(llm, system_prompt="expert system")

        response = await expert.consult(
            question="What is the clinical context?",
            findings={"glucose": 148, "bmi": 33.6},
        )

        assert isinstance(response, MedicalExpertResponse)
        assert "glucose" in response.reasoning
        assert len(response.citations) == 1
        assert llm.messages is not None

    def test_builds_stub_when_no_expert_env_is_configured(self, monkeypatch):
        monkeypatch.delenv("MEDICAL_EXPERT_PRIMARY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        expert = build_medical_expert_agent()

        assert isinstance(expert, StubMedicalExpert)
