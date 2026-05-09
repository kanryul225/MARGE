"""Tests for the BeeAI tool adapter.

The adapter wraps a Python callable + Pydantic input schema as a BeeAI Tool,
preserving name and description and routing input through the schema.
`local_tools_as_beeai(bundle)` returns the five orchestrator-local tools as
BeeAI Tools with the same enforcer wired in.
"""

import asyncio

from pydantic import BaseModel, Field

from apps.orchestrator.agent import build_bundle
from apps.orchestrator.tools._adapter import local_tools_as_beeai, to_beeai_tool


class _SampleInput(BaseModel):
    x: int = Field(description="a number")


class TestToBeeaiTool:
    def test_preserves_name_and_description(self):
        def fn(input_obj: _SampleInput) -> dict:
            return {"x": input_obj.x}

        bt = to_beeai_tool(
            fn, name="sample", description="sample tool", input_schema=_SampleInput
        )
        assert bt.name == "sample"
        assert bt.description == "sample tool"

    def test_returns_a_beeai_tool(self):
        from beeai_framework.tools import Tool

        def fn(input_obj: _SampleInput) -> dict:
            return {"x": input_obj.x}

        bt = to_beeai_tool(
            fn, name="sample", description="d", input_schema=_SampleInput
        )
        assert isinstance(bt, Tool)


class TestLocalToolsAsBeeai:
    def test_returns_five_tools(self):
        bundle = build_bundle()
        tools = local_tools_as_beeai(bundle)
        assert len(tools) == 5

    def test_tool_names_match_expected_set(self):
        bundle = build_bundle()
        tools = local_tools_as_beeai(bundle)
        names = {t.name for t in tools}
        assert names == {
            "get_patient_history",
            "consult_medical_expert",
            "final_report",
            "abstain",
            "ask_user_back",
        }

    def test_calling_get_patient_history_records_in_bundle_enforcer(self):
        """The wrapped BeeAI tool must use the bundle's enforcer instance."""
        bundle = build_bundle()
        tools = {t.name: t for t in local_tools_as_beeai(bundle)}

        get_history = tools["get_patient_history"]
        # BeeAI tools are awaitable; invoke synchronously for the test.
        asyncio.get_event_loop().run_until_complete(
            get_history.run({"handle": "seed-001"})
        )
        assert bundle.enforcer.has_called("get_patient_history")
