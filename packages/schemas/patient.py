"""Schemas for patient records.

A `PatientRecord` is the unified representation that the orchestrator works
with, regardless of whether the data came from the seeded SQLite DB or a
user-uploaded CSV. The `patient_data_mcp_server` is responsible for resolving
both source types into this same shape.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


Sex = Literal["female", "male", "other"]


class ClinicalFeature(BaseModel):
    """A single named feature from a patient record (typed, optionally with units)."""

    name: str
    value: float | int | str | bool | None
    unit: str | None = None
    measured_at: date | None = None


class PatientRecord(BaseModel):
    """Unified patient representation across all data sources.

    The `features` dict is the surface that ML models read from — keys must
    match the input schema of the model the orchestrator wants to call. The
    orchestrator is responsible for mapping `features` into a model's input
    shape (e.g., dropping fields the model does not need).
    """

    handle: str = Field(
        description="Source-prefixed ID: 'seed-NN' for seeded patients, 'upload-XXXX' for uploads"
    )
    age: int | None = None
    sex: Sex | None = None
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Flat name->value map for ML model consumption. Keys should match dataset feature names.",
    )
    notes: str | None = Field(
        default=None, description="Free-text clinical notes attached to the patient"
    )
