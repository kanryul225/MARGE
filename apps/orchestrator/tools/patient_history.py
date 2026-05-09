"""Local tool: fetch a patient record from the patient data source."""

from collections.abc import Callable

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource

TOOL_NAME = "get_patient_history"
TOOL_DESCRIPTION = (
    "Fetch a patient record by handle (e.g., 'seed-001' or 'upload-XXXX'). "
    "Returns the unified PatientRecord with demographics and a flat feature dict "
    "that downstream ML tools consume."
)


class ToolInput(BaseModel):
    handle: str = Field(
        description="Source-prefixed patient ID, e.g. 'seed-001' or 'upload-XXXX'."
    )


def make_patient_history(
    source: PatientSource,
    enforcer: ProtocolEnforcer,
) -> Callable[..., PatientRecord]:
    """Build the get_patient_history tool bound to a specific source + enforcer."""

    def get_patient_history(handle: str) -> PatientRecord:
        enforcer.record(TOOL_NAME)
        return source.resolve(handle)

    get_patient_history.__doc__ = TOOL_DESCRIPTION
    return get_patient_history
