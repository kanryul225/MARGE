"""Local tool: request_more_info — terminal ML catalog + feature inventory.

Use this when a disease/condition might map to a local ML predictor, but the
orchestrator needs to know whether such a model exists and which feature data
is still required before calling `predict_*`.

This is a terminal because it ends the agent loop and waits for the user's
next message. It is NOT gated — the orchestrator may request more info at
any point in the conversation, including before any expert consultation.
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "request_more_info"
TOOL_DESCRIPTION = (
    "Terminal: check the local ML catalog for a disease/condition, determine "
    "whether a matching predict_* model exists, compare the model's required "
    "features against known patient data, and ask the user only for the missing "
    "feature values needed to run the relevant ML model. Always allowed "
    "(no protocol prerequisites)."
)


_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "predict_diabetes_risk": (
        "diabetes",
        "diabetic",
        "dysglycemia",
        "hyperglycemia",
        "blood sugar",
        "glucose",
        "당뇨",
        "혈당",
        "고혈당",
    ),
    "predict_breast_cancer_malignancy": (
        "breast cancer",
        "breast",
        "malignancy",
        "malignant",
        "tumor",
        "tumour",
        "fna",
        "biopsy",
        "mammography",
        "유방암",
        "유방",
        "종양",
        "악성",
        "조직검사",
    ),
}

_UNIT_HINTS: dict[str, str] = {
    "preg": "count",
    "plas": "mg/dL",
    "pres": "mmHg",
    "skin": "mm",
    "insu": "mu U/mL",
    "mass": "kg/m^2",
    "pedi": "score",
    "age": "years",
}

_FIELD_TYPE_HINTS: dict[str, str] = {
    "preg": "number",
    "plas": "number",
    "pres": "number",
    "skin": "number",
    "insu": "number",
    "mass": "number",
    "pedi": "number",
    "age": "number",
}


class NeededField(BaseModel):
    name: str = Field(description="Short name of the field (e.g., 'HbA1c', 'family_history_diabetes').")
    why: str = Field(description="One-line clinical reason this field would shift the analysis.")
    field_type: Literal["number", "text", "category", "yes_no"] = Field(
        default="text",
        description="Hint to the UI for input rendering.",
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit if numeric (e.g., 'mg/dL', 'kg/m^2').",
    )


class ToolInput(BaseModel):
    target_condition: str | None = Field(
        default=None,
        description=(
            "Disease/condition to check against the local ML catalog, e.g. "
            "'diabetes', 'breast cancer', '유방암'."
        ),
    )
    known_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Patient values already known from the conversation or patient record. "
            "Keys can be model feature names or common aliases such as glucose, BMI, age."
        ),
    )
    needed: list[NeededField] = Field(
        default_factory=list,
        description=(
            "Optional manual fallback list. Prefer target_condition + known_features "
            "so the tool can compute missing ML features from the local catalog."
        ),
    )
    rationale: str = Field(
        description=(
            "One- or two-sentence framing for the user explaining why these "
            "specific data points matter for the analysis."
        ),
    )
    max_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of matching local ML models to return.",
    )


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value.replace("_", " ") if ch.isalnum())


def _field_metadata(model: Any, field_name: str) -> dict[str, Any]:
    field = model.input_schema.model_fields[field_name]
    extra = field.json_schema_extra or {}
    return {
        "label": extra.get("label") or field_name.replace("_", " "),
        "detail": field.description or f"Feature measurement: {field_name}.",
        "aliases": extra.get("aliases", []),
    }


def _known_feature_names(model: Any, known_features: dict[str, Any]) -> set[str]:
    known_lookup = {
        _normalize_key(key): value
        for key, value in known_features.items()
        if value is not None and value != ""
    }
    known: set[str] = set()
    for field_name in model.input_schema.model_fields:
        metadata = _field_metadata(model, field_name)
        candidate_names = [
            field_name,
            field_name.replace("_", " "),
            metadata["label"],
            *metadata["aliases"],
        ]
        if any(_normalize_key(str(name)) in known_lookup for name in candidate_names):
            known.add(field_name)
    return known


def _needed_field_for(model: Any, field_name: str) -> dict[str, Any]:
    metadata = _field_metadata(model, field_name)
    return {
        "name": field_name,
        "label": metadata["label"],
        "why": metadata["detail"],
        "field_type": _FIELD_TYPE_HINTS.get(field_name, "number"),
        "unit": _UNIT_HINTS.get(field_name),
        "aliases": metadata["aliases"],
    }


def _match_score(model: Any, target_condition: str) -> int:
    target = target_condition.lower()
    haystack = " ".join(
        [
            model.name,
            getattr(model.metadata, "description", ""),
            getattr(model.metadata, "trained_on", ""),
            " ".join(_MODEL_ALIASES.get(model.name, ())),
        ]
    ).lower()
    score = 0
    if target in haystack:
        score += 10
    for token in target.replace("/", " ").replace("-", " ").split():
        if len(token) >= 3 and token in haystack:
            score += 2
    for alias in _MODEL_ALIASES.get(model.name, ()):
        if alias.lower() in target or target in alias.lower():
            score += 8
    return score


def _catalog_matches(target_condition: str, max_models: int) -> list[Any]:
    from services.ml_mcp_server.registry import discover_models

    scored = [
        (_match_score(model, target_condition), model)
        for model in discover_models()
    ]
    scored = [(score, model) for score, model in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [model for _, model in scored[:max_models]]


def _available_model_summaries() -> list[dict[str, Any]]:
    from services.ml_mcp_server.registry import discover_models

    return [
        {
            "model": model.name,
            "description": model.metadata.description,
            "feature_count": model.metadata.feature_count,
        }
        for model in discover_models()
    ]


def make_request_more_info(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def request_more_info(
        target_condition: str | None = None,
        known_features: dict[str, Any] | None = None,
        needed: list[dict[str, Any]] | None = None,
        rationale: str | None = None,
        max_models: int = 3,
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)

        if not target_condition:
            return {
                "needs_more_info": True,
                "catalog_checked": False,
                "model_available": None,
                "target_condition": None,
                "matched_models": [],
                "needed": needed or [],
                "rationale": rationale or "Additional information is needed before analysis can proceed.",
            }

        known_features = known_features or {}
        matches = _catalog_matches(target_condition, max_models=max_models)

        if not matches:
            return {
                "needs_more_info": False,
                "catalog_checked": True,
                "model_available": False,
                "target_condition": target_condition,
                "matched_models": [],
                "available_models": _available_model_summaries(),
                "needed": [],
                "rationale": (
                    rationale
                    or f"No local ML model matches '{target_condition}'. Use expert reasoning or abstain."
                ),
            }

        model_payloads: list[dict[str, Any]] = []
        combined_needed: list[dict[str, Any]] = []
        seen_needed: set[str] = set()
        for model in matches:
            field_names = list(model.input_schema.model_fields.keys())
            present = _known_feature_names(model, known_features)
            missing = [name for name in field_names if name not in present]
            missing_payload = [_needed_field_for(model, name) for name in missing]
            for item in missing_payload:
                key = item["name"]
                if key not in seen_needed:
                    seen_needed.add(key)
                    combined_needed.append(item)
            model_payloads.append(
                {
                    "model": model.name,
                    "description": model.metadata.description,
                    "trained_on": model.metadata.trained_on,
                    "feature_count": model.metadata.feature_count,
                    "can_run_now": len(missing) == 0,
                    "available_features": sorted(present),
                    "missing_features": missing_payload,
                    "required_features": [
                        _needed_field_for(model, name) for name in field_names
                    ],
                }
            )

        return {
            "needs_more_info": True,
            "catalog_checked": True,
            "model_available": True,
            "target_condition": target_condition,
            "matched_models": model_payloads,
            "needed": combined_needed,
            "rationale": (
                rationale
                or (
                    f"Found {len(model_payloads)} local ML model(s) for '{target_condition}'. "
                    "Please provide the missing feature values so the orchestrator can run the relevant predict_* tool."
                )
            ),
        }

    request_more_info.__doc__ = TOOL_DESCRIPTION
    return request_more_info
