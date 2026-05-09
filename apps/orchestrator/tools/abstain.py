"""Local tool: abstain — decline to give a clinical recommendation.

Always allowed (no protocol prerequisites). Prefer when ML predictions
conflict irresolvably or the medical expert flags the data as unreliable.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "abstain"
TOOL_DESCRIPTION = (
    "Decline to provide a clinical recommendation, advising the user to seek "
    "professional medical care. Use when ML predictions conflict irresolvably, "
    "the medical expert flags the data as unreliable, or you cannot answer with "
    "sufficient confidence."
)


class ToolInput(BaseModel):
    reason: str = Field(
        description="Why a reliable answer cannot be given (e.g., 'ML predictions conflict and expert flagged data quality')."
    )


def make_abstain(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, str | bool]]:
    def abstain(reason: str) -> dict[str, str | bool]:
        enforcer.check_can_abstain()
        enforcer.record(TOOL_NAME)
        return {"abstained": True, "reason": reason}

    abstain.__doc__ = TOOL_DESCRIPTION
    return abstain
