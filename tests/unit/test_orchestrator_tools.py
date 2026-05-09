"""Tests for the orchestrator's local tool factories.

Three local tools (architecture.md §2 + the single-terminal refactor):
- consult_expert    — dispatches to the medical_expert_agent + records call
- patient_history   — dispatches to a PatientSource + records call
- final_report      — only path to a user-facing answer; gated by
                       ProtocolEnforcer.check_finalize() as the code-side
                       defensive backstop (LLM-side gating is via
                       MARGEProtocolRequirement).

`abstain` and `ask_user_back` are deliberately not separate tools any more
— their content is expressed in `final_report`'s `response` field.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from packages.schemas.patient import PatientRecord
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


class TestConsultExpertTool:
    def test_returns_medical_expert_response(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        response = consult(question="What does this suggest?", findings={"a": 1})
        assert isinstance(response, MedicalExpertResponse)

    def test_records_consult_medical_expert_call(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        consult(question="?", findings={})
        assert enforcer.has_called("consult_medical_expert")


class TestPatientHistoryTool:
    def test_returns_patient_record(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        record = get_history(handle="seed-001")
        assert isinstance(record, PatientRecord)
        assert record.handle == "seed-001"

    def test_records_get_patient_history_call(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        get_history(handle="seed-001")
        assert enforcer.has_called("get_patient_history")

    def test_propagates_keyerror_for_unknown_handle(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        with pytest.raises(KeyError):
            get_history(handle="seed-9999")


class TestFinalReportTool:
    def test_blocks_when_no_ml_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="ML model"):
            final(response="anything")

    def test_blocks_when_no_expert_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="expert"):
            final(response="anything")

    def test_succeeds_when_workflow_sequence_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        result = final(response="High-confidence findings — refer for biopsy.")
        assert result == {"response": "High-confidence findings — refer for biopsy."}

    def test_records_final_report_call(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_diabetes_risk")
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        final(response="...")
        assert enforcer.has_called("final_report")
