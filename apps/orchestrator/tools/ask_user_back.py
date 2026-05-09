"""Local tool: ask the user for additional information before answering.

Always allowed (no protocol prerequisites). Prefer when one missing feature
would meaningfully shift the ML predictions — i.e., where the information
gain is high.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "ask_user_back"
TOOL_DESCRIPTION = (
    "Request additional information from the user before answering. Use when one "
    "missing feature would meaningfully shift the ML predictions — i.e., where the "
    "information gain is high."
)


class ToolInput(BaseModel):
    missing_info: list[str] = Field(
        description="List of items to request from the user (e.g., recent labs, family history, current medications)."
    )


def make_ask_user_back(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def ask_user_back(missing_info: list[str]) -> dict[str, Any]:
        enforcer.check_can_ask_user_back()
        enforcer.record(TOOL_NAME)
        return {"asking_user_back": True, "missing_info": missing_info}

    ask_user_back.__doc__ = TOOL_DESCRIPTION
    return ask_user_back
