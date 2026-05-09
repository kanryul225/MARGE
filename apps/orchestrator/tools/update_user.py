"""Local tool: update_user — non-terminal mid-flow message to the user.

Use this whenever you want to say something to the user without ending
the turn. Examples:
- Greeting / acknowledging the question.
- Sharing intermediate progress ("Now consulting the medical expert about
  the differential…").
- Explaining what you're about to run ("Going to check the diabetes risk
  model with the values you provided.").
- Paraphrasing the medical expert's reasoning before further work.

Multiple update_user calls per turn are encouraged. The agent loop continues
after each call — only `clinical_report`, `abstain`, or `request_more_info`
end the turn.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "update_user"
TOOL_DESCRIPTION = (
    "Send a natural-language message to the user mid-flow. THIS DOES NOT END "
    "THE TURN — the agent continues working afterwards. Use freely for "
    "greetings, progress updates, and explanations of what you're about to do. "
    "When you actually finish, use clinical_report / abstain / request_more_info."
)


class ToolInput(BaseModel):
    text: str = Field(
        description=(
            "Message to display to the user. Conversational tone. Keep each "
            "update focused — split long updates into multiple update_user calls "
            "if helpful."
        )
    )


def make_update_user(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def update_user(text: str) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {"text": text}

    update_user.__doc__ = TOOL_DESCRIPTION
    return update_user
