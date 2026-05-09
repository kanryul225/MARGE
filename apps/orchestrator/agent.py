"""BeeAI orchestrator assembly.

Two layers:

`build_bundle()` returns the deterministic dependencies — enforcer, stubs,
local tools, system prompt — exactly as architecture.md §4 specifies. This
is what every test reaches for; it has no LLM dependency.

`build_orchestrator_agent(bundle, llm)` wraps the bundle into a fully
configured BeeAI `ToolCallingAgent`: local tools converted to BeeAI Tools,
ML tools auto-discovered from the in-process `ml-models` MCP server, system
prompt mounted on the agent's metadata, memory attached. The LLM is passed
in by the caller so tests can inject a fake one.

Real-LLM smoke verification lives in `scripts/orchestrator_smoke.py`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.ask_user_back import make_ask_user_back
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource

if TYPE_CHECKING:
    from beeai_framework.agents.tool_calling import ToolCallingAgent
    from beeai_framework.backend.chat import ChatModel

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


@dataclass
class OrchestratorBundle:
    """Deterministic dependencies of the orchestrator (no LLM)."""

    enforcer: ProtocolEnforcer
    system_prompt: str
    local_tools: dict[str, object]
    patient_source: PatientSource


def build_bundle(
    patient_source: PatientSource | None = None,
) -> OrchestratorBundle:
    """Build the orchestrator's deterministic dependencies."""
    enforcer = ProtocolEnforcer()
    expert = StubMedicalExpert()
    source = patient_source or SqlitePatientSource()

    local_tools = {
        "get_patient_history": make_patient_history(source, enforcer),
        "consult_medical_expert": make_consult_expert(expert, enforcer),
        "final_report": make_final_report(enforcer),
        "abstain": make_abstain(enforcer),
        "ask_user_back": make_ask_user_back(enforcer),
    }

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=_SYSTEM_PROMPT_PATH.read_text(),
        local_tools=local_tools,
        patient_source=source,
    )


async def build_orchestrator_agent(
    bundle: OrchestratorBundle,
    llm: "ChatModel",
) -> "RequirementAgent":
    """Assemble a fully wired BeeAI RequirementAgent.

    - Local tools are converted from the bundle's factory closures
    - ML tools are auto-discovered from the in-process `ml-models` MCP server
    - The system prompt is the agent's `instructions`
    - Memory is `UnconstrainedMemory` (single-user, single-session demo)

    Future work: declare the protocol invariants (ML+expert before final_report)
    as BeeAI `Requirement`s rather than enforcing them inside the tool body —
    that lifts the constraint into the agent loop's planning surface.
    """
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory

    from apps.orchestrator.mcp_discovery import discover_ml_mcp_tools
    from apps.orchestrator.tools._adapter import local_tools_as_beeai

    local_tools = local_tools_as_beeai(bundle)
    ml_tools = await discover_ml_mcp_tools()

    return RequirementAgent(
        llm=llm,
        memory=UnconstrainedMemory(),
        tools=[*local_tools, *ml_tools],
        name="MARGE Orchestrator",
        description=(
            "Clinical ML head researcher: orchestrates ML tools and a medical "
            "expert sub-agent."
        ),
        instructions=bundle.system_prompt,
    )
