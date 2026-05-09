"""BeeAI orchestrator assembly.

Two layers:

`build_bundle()` returns the deterministic dependencies — enforcer, stubs,
local tools, system prompt — exactly as architecture.md §4 specifies. This
is what every test reaches for; it has no LLM dependency.

`orchestrator_agent(bundle, llm)` is an async context manager that:
- Opens an in-process MCP Client against the `ml-models` server
- Discovers ML tools via the live session
- Wraps local tools with the BeeAI adapter
- Yields a fully wired `RequirementAgent` (5 local + N MCP tools)
- Closes the MCP session on exit

The session must stay open for the lifetime of the agent run — `MCPTool`
holds a reference to the session and tearing it down before agent.run()
makes every tool call fail with `ToolError`.

Real-LLM smoke verification: `scripts/orchestrator_smoke.py`.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.requirements.marge_protocol import (
    build_marge_protocol_requirement,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource

if TYPE_CHECKING:
    from beeai_framework.agents.requirement import RequirementAgent
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
    }

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=_SYSTEM_PROMPT_PATH.read_text(),
        local_tools=local_tools,
        patient_source=source,
    )


@asynccontextmanager
async def orchestrator_agent(
    bundle: OrchestratorBundle,
    llm: "ChatModel",
) -> AsyncIterator["RequirementAgent"]:
    """Build and yield a fully wired RequirementAgent; close MCP session on exit.

    Why this is a context manager: BeeAI's MCPTool holds a reference to the
    MCP ClientSession. If we close the session before the agent finishes,
    every MCP tool call inside agent.run() raises ToolError. Keeping the
    session open for the agent's lifetime is the simplest correct fix.

    Usage:
        async with orchestrator_agent(bundle, llm) as agent:
            result = await agent.run("…")
    """
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory
    from fastmcp import Client

    from apps.orchestrator.tools._adapter import local_tools_as_beeai
    from services.ml_mcp_server.server import build_server

    local_tools = local_tools_as_beeai(bundle)
    mcp_server = build_server()

    async with Client(mcp_server) as mcp_client:
        from beeai_framework.tools.mcp import MCPTool

        ml_tools = await MCPTool.from_client(mcp_client.session)

        agent = RequirementAgent(
            llm=llm,
            memory=UnconstrainedMemory(),
            tools=[*local_tools, *ml_tools],
            requirements=[build_marge_protocol_requirement()],
            name="MARGE Orchestrator",
            description=(
                "Clinical ML head researcher: orchestrates ML tools and a "
                "medical expert sub-agent."
            ),
            instructions=bundle.system_prompt,
            # Disable BeeAI's automatic FinalAnswerTool — our `final_report`
            # is the gated terminal tool from architecture.md §2 and must
            # be the only path to a user answer.
            final_answer_as_tool=False,
        )

        # Hook the enforcer onto MCP tools only. Local tools already
        # record themselves via the factory closures (test-covered).
        # ML predictions go through MCP, so without this hook the
        # finalize gate would never see a `predict_*` call and would
        # incorrectly block final_report.
        def _make_recorder(tool_name: str):
            def _record(data, event) -> None:
                # Fire only on the start event for each tool; using "*" so
                # it matches across BeeAI's emitter namespacing.
                if getattr(event, "name", None) == "start":
                    bundle.enforcer.record(tool_name)
            return _record

        for t in ml_tools:
            t.emitter.match("*", _make_recorder(t.name))

        yield agent
