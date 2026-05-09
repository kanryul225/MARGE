"""Tests for the per-tool Pydantic input schemas.

Each local tool module exports a `ToolInput` Pydantic class plus `TOOL_NAME`
and `TOOL_DESCRIPTION` module constants. These are consumed by the BeeAI
adapter to build properly-typed BeeAI Tools.
"""

import pytest
from pydantic import ValidationError

from apps.orchestrator.tools import (
    abstain as ab_mod,
    ask_user_back as au_mod,
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

    def test_accepts_full_payload(self):
        obj = fr_mod.ToolInput(
            summary="s", recommendation="r", confidence_note="c"
        )
        assert obj.summary == "s"
        assert obj.confidence_note == "c"

    def test_confidence_note_optional(self):
        obj = fr_mod.ToolInput(summary="s", recommendation="r")
        assert obj.confidence_note is None


class TestAbstainSchema:
    def test_constants_exposed(self):
        assert ab_mod.TOOL_NAME == "abstain"
        assert ab_mod.TOOL_DESCRIPTION

    def test_accepts_reason(self):
        obj = ab_mod.ToolInput(reason="no data")
        assert obj.reason == "no data"


class TestAskUserBackSchema:
    def test_constants_exposed(self):
        assert au_mod.TOOL_NAME == "ask_user_back"
        assert au_mod.TOOL_DESCRIPTION

    def test_accepts_missing_info_list(self):
        obj = au_mod.ToolInput(missing_info=["a", "b"])
        assert obj.missing_info == ["a", "b"]
