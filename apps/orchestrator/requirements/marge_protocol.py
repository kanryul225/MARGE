"""BeeAI Requirements encoding the MARGE workflow.

The workflow is:

    consult_medical_expert -> predict_* -> consult_medical_expert -> final_report

Two structural layers enforce this:

- Each discovered `predict_*` tool is disallowed until at least one expert
  pre-consult has succeeded.
- `final_report` is disallowed until the successful trajectory contains an
  expert pre-consult, at least one ML prediction after that consult, and an
  expert post-consult after the ML result.
"""

from collections.abc import Iterable
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


def has_pre_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded before a successful ML prediction."""
    seen_expert = False
    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            seen_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_expert:
            return True
    return False


def has_post_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded after a successful ML prediction."""
    seen_ml = False
    for name in _successful_tool_names(state):
        if name.startswith(_ML_PREDICTION_PREFIX):
            seen_ml = True
        elif name == _EXPERT_TOOL_NAME and seen_ml:
            return True
    return False


def has_expert_ml_expert_sequence(state: Any) -> bool:
    """True if the successful trajectory contains expert -> ML -> expert."""
    seen_pre_expert = False
    seen_ml_after_pre_expert = False

    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            if seen_ml_after_pre_expert:
                return True
            seen_pre_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_pre_expert:
            seen_ml_after_pre_expert = True

    return False


def _build_ml_preconsult_requirement(tool_name: str) -> ConditionalRequirement:
    return ConditionalRequirement(
        target=tool_name,
        custom_checks=[has_consulted_expert],
        only_success_invocations=True,
        reason=(
            f"{tool_name} requires a successful consult_medical_expert pre-consult "
            "before any ML prediction tool can run."
        ),
    )


def build_marge_protocol_requirement() -> ConditionalRequirement:
    """Construct the terminal Requirement that gates `final_report`.

    - `custom_checks` keeps `final_report` disallowed until the successful
      trajectory contains expert -> ML -> expert.
    - `min_invocations=1` keeps `prevent_stop=True` until `final_report` has
      been called at least once, so the agent cannot terminate without
      producing a final report.
    """
    return ConditionalRequirement(
        target=_TERMINAL_TOOL_NAME,
        custom_checks=[has_expert_ml_expert_sequence],
        min_invocations=1,
        only_success_invocations=True,
        reason=(
            "final_report requires a successful workflow sequence: "
            "consult_medical_expert -> predict_* -> consult_medical_expert."
        ),
    )


def build_marge_protocol_requirements(
    ml_tool_names: Iterable[str] = (),
) -> list[ConditionalRequirement]:
    """Build all structural requirements for the current tool surface."""
    return [
        *[
            _build_ml_preconsult_requirement(name)
            for name in ml_tool_names
            if name.startswith(_ML_PREDICTION_PREFIX)
        ],
        build_marge_protocol_requirement(),
    ]
