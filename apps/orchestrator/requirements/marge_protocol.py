"""BeeAI Requirement encoding architecture.md §2.

Single rule on the orchestrator's only terminal tool, `final_report`:

- It is `allowed` only after the trajectory contains at least one
  predict_* tool call AND at least one consult_medical_expert call,
  in any order.
- The agent's stop is `prevent`-ed until `final_report` has been
  called at least once.

The two `custom_checks` look at the entire successful step trajectory
without any ordering constraint between predict_* and consult_*. Multiple
calls in either direction are fine; abstention or follow-up questions
are expressed in `final_report`'s natural-language `response` field
rather than via separate tools.
"""

from typing import Any

from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)


_TERMINAL_TOOL_NAME = "final_report"
_ML_PREDICTION_PREFIX = "predict_"
_EXPERT_TOOL_NAME = "consult_medical_expert"


def _successful_tool_names(state: Any) -> list[str]:
    return [
        s.tool.name
        for s in state.steps
        if s.tool is not None and not s.error and getattr(s.tool, "name", None)
    ]


def has_any_ml_prediction(state: Any) -> bool:
    """True if any predict_* tool has succeeded in the trajectory."""
    return any(name.startswith(_ML_PREDICTION_PREFIX) for name in _successful_tool_names(state))


def has_consulted_expert(state: Any) -> bool:
    """True if consult_medical_expert has succeeded in the trajectory."""
    return _EXPERT_TOOL_NAME in _successful_tool_names(state)


def build_marge_protocol_requirement() -> ConditionalRequirement:
    """Construct the single Requirement that gates `final_report`.

    - `custom_checks` keeps `final_report` disallowed until both checks pass.
    - `min_invocations=1` keeps `prevent_stop=True` until `final_report` has
      been called at least once — so the agent cannot terminate without
      producing a final report.
    """
    return ConditionalRequirement(
        target=_TERMINAL_TOOL_NAME,
        custom_checks=[has_any_ml_prediction, has_consulted_expert],
        min_invocations=1,
        only_success_invocations=True,
        reason=(
            "final_report requires at least one ML prediction (predict_*) and "
            "one consult_medical_expert call to have appeared in the trajectory."
        ),
    )
