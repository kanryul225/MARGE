"""Integration: code-side protocol flow without LLM (single-terminal version).

Simulates the orchestrator's tool-call sequence (recording each step in
the ProtocolEnforcer) and verifies the defensive backstop blocks
`final_report` until ML and expert have appeared in the trajectory.

Order between predict_* and consult_medical_expert is intentionally
unconstrained — both orderings pass.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


@pytest.fixture
def deps():
    enforcer = ProtocolEnforcer()
    return {
        "enforcer": enforcer,
        "history": make_patient_history(SqlitePatientSource(), enforcer),
        "consult": make_consult_expert(StubMedicalExpert(), enforcer),
        "final": make_final_report(enforcer),
    }


def test_happy_path_ml_before_expert(deps):
    deps["history"](handle="seed-001")
    deps["enforcer"].record("predict_breast_cancer_malignancy")  # simulated MCP
    deps["consult"](
        question="What does this prediction suggest?",
        findings={"prediction": "malignant", "confidence": 0.989},
    )
    result = deps["final"](response="High-confidence finding — refer for biopsy.")
    assert result == {"response": "High-confidence finding — refer for biopsy."}


def test_happy_path_expert_before_ml(deps):
    """Expert can be consulted first — order is free."""
    deps["history"](handle="seed-001")
    deps["consult"](question="Which model should we run?", findings={})
    deps["enforcer"].record("predict_diabetes_risk")
    result = deps["final"](response="Risk profile suggests follow-up labs.")
    assert "response" in result


def test_blocked_when_skipping_ml(deps):
    deps["consult"](question="?", findings={})
    with pytest.raises(ProtocolViolation, match="ML model"):
        deps["final"](response="anything")


def test_blocked_when_skipping_expert(deps):
    deps["enforcer"].record("predict_diabetes_risk")
    with pytest.raises(ProtocolViolation, match="expert"):
        deps["final"](response="anything")


def test_trajectory_records_full_sequence(deps):
    deps["history"](handle="seed-001")
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](question="?", findings={})
    deps["final"](response="ok")

    assert deps["enforcer"].trajectory == (
        "get_patient_history",
        "predict_breast_cancer_malignancy",
        "consult_medical_expert",
        "final_report",
    )
