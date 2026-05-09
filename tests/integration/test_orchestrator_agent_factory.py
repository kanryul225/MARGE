"""Integration: assemble the BeeAI ToolCallingAgent from the bundle + MCP tools.

`build_orchestrator_agent(bundle, llm)` discovers ML tools via MCP, converts
the bundle's local tools to BeeAI Tools, and returns a fully wired
`ToolCallingAgent` with the system prompt and memory configured.

We pass a fake ChatModel (no real LLM call) — these tests only verify that
the agent has the expected tool surface and configuration. Live LLM calls
happen in `scripts/orchestrator_smoke.py`.
"""

import pytest

from apps.orchestrator.agent import build_bundle, build_orchestrator_agent


class _FakeChatModel:
    """Stand-in for BeeAI ChatModel — never invoked, just stored on the agent."""

    model_id = "fake-model"
    provider_id = "fake"
    parameters = None
    tool_choice_support = ["auto", "single"]
    middlewares = ()


@pytest.mark.asyncio
async def test_returns_requirement_agent():
    from beeai_framework.agents.requirement import RequirementAgent

    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=_FakeChatModel())
    assert isinstance(agent, RequirementAgent)


@pytest.mark.asyncio
async def test_agent_has_local_tools_plus_ml_tools():
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=_FakeChatModel())
    # _tools is the stable internal accessor BeeAI's runner uses.
    tool_names = {t.name for t in agent._tools}
    expected_local = {
        "get_patient_history",
        "consult_medical_expert",
        "final_report",
        "abstain",
        "ask_user_back",
    }
    expected_ml = {"predict_breast_cancer_malignancy", "predict_diabetes_risk"}
    assert expected_local <= tool_names, f"missing local tools: {expected_local - tool_names}"
    assert expected_ml <= tool_names, f"missing ML tools: {expected_ml - tool_names}"


@pytest.mark.asyncio
async def test_agent_has_seven_tools_total():
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=_FakeChatModel())
    # 5 local + 2 MCP-discovered ML = 7
    assert len(agent._tools) == 7


@pytest.mark.asyncio
async def test_agent_uses_provided_llm():
    fake = _FakeChatModel()
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=fake)
    assert agent._llm is fake


@pytest.mark.asyncio
async def test_system_prompt_marker_present_in_bundle():
    """The system prompt is fed to BeeAI's `instructions` parameter and merged
    into RequirementAgent's prompt template at run time. We verify that the
    bundle (the source of truth) carries the role marker — BeeAI's wiring of
    `instructions` into its template is its own responsibility."""
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=_FakeChatModel())
    assert "ML Head Researcher" in bundle.system_prompt
    # And the agent has metadata derived from our build call:
    assert agent.meta.name == "MARGE Orchestrator"
