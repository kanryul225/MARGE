"""Local tool: emit the final report to the user.

Gated by ProtocolEnforcer: fails with ProtocolViolation unless an ML
prediction tool AND consult_medical_expert have been called in the
trajectory. See architecture.md §2.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "final_report"
TOOL_DESCRIPTION = (
    "Emit the final clinical report to the user. THIS IS THE ONLY PATH TO A "
    "USER-FACING ANSWER. Fails unless at least one ML prediction tool and "
    "consult_medical_expert have already been called in this turn."
)


class ToolInput(BaseModel):
    summary: str = Field(description="One-paragraph summary of the case and findings.")
    recommendation: str = Field(description="Action the clinician should consider.")
    confidence_note: str | None = Field(
        default=None, description="Optional confidence / agreement note."
    )


def make_final_report(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def final_report(
        summary: str,
        recommendation: str,
        confidence_note: str | None = None,
    ) -> dict[str, Any]:
        enforcer.check_finalize()
        enforcer.record(TOOL_NAME)
        return {
            "summary": summary,
            "recommendation": recommendation,
            "confidence_note": confidence_note,
        }

    final_report.__doc__ = TOOL_DESCRIPTION
    return final_report
