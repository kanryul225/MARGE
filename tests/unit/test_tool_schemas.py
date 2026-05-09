"""Tests for the per-tool Pydantic input schemas.

Three local tools after the single-terminal refactor: each module exports
a `ToolInput` Pydantic class plus `TOOL_NAME` and `TOOL_DESCRIPTION`
module constants. These are consumed by the BeeAI adapter to build
properly-typed BeeAI Tools.
"""

import pytest
from pydantic import ValidationError

from apps.orchestrator.tools import (
    consult_expert as ce_mod,
    final_report as fr_mod,
    patient_history as ph_mod,
)


class TestPatientHistorySchema:
    def test_constants_exposed(self):
        assert ph_mod.TOOL_NAME == "get_patient_history"
        assert ph_mod.TOOL_DESCRIPTION

    def test_accepts_handle(self):
        obj = ph_mod.ToolInput(handle="seed-001")
        assert obj.handle == "seed-001"

    def test_rejects_missing_handle(self):
        with pytest.raises(ValidationError):
            ph_mod.ToolInput()


class TestConsultExpertSchema:
    def test_constants_exposed(self):
        assert ce_mod.TOOL_NAME == "consult_medical_expert"
        assert ce_mod.TOOL_DESCRIPTION

    def test_accepts_question_and_findings(self):
        obj = ce_mod.ToolInput(question="why?", findings={"a": 1})
        assert obj.question == "why?"
        assert obj.findings == {"a": 1}

    def test_findings_defaults_to_empty(self):
        obj = ce_mod.ToolInput(question="?")
        assert obj.findings == {}


class TestFinalReportSchema:
    def test_constants_exposed(self):
        assert fr_mod.TOOL_NAME == "final_report"
        assert fr_mod.TOOL_DESCRIPTION

    def test_accepts_response(self):
        obj = fr_mod.ToolInput(response="Recommendation: X")
        assert obj.response == "Recommendation: X"

    def test_rejects_missing_response(self):
        with pytest.raises(ValidationError):
            fr_mod.ToolInput()
