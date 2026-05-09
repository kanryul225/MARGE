"""Diagnostic: progressively exercise the orchestrator agent to find where it hangs.

Each phase prints before running so we know exactly which step blocks.

Run unbuffered: `uv run python -u scripts/diag_agent.py`
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def step(name: str) -> None:
    print(f"\n[diag] >>> {name}", flush=True)


async def main() -> None:
    step("import beeai modules")
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.backend.message import UserMessage
    from beeai_framework.memory import UnconstrainedMemory

    step("build llm (NVIDIA Qwen3.5-397B)")
    from packages.llm_provider.client import build_chat_model_for_role
    from packages.llm_provider.settings import Role

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    print(f"      llm: {llm.model_id} via {llm.provider_id}", flush=True)

    step("direct LLM hello")
    out = await llm.run([UserMessage("Reply with exactly: HELLO_OK")])
    text = out.get_text_content() if hasattr(out, "get_text_content") else str(out)
    print(f"      <- {text!r}", flush=True)

    step("build orchestrator bundle (no LLM)")
    from apps.orchestrator.agent import build_bundle

    bundle = build_bundle()
    print(f"      tools: {sorted(bundle.local_tools.keys())}", flush=True)

    step("convert local tools to BeeAI Tools")
    from apps.orchestrator.tools._adapter import local_tools_as_beeai

    local_tools = local_tools_as_beeai(bundle)
    print(f"      local BeeAI tools: {[t.name for t in local_tools]}", flush=True)

    step("discover ML MCP tools")
    from apps.orchestrator.mcp_discovery import discover_ml_mcp_tools

    ml_tools = await discover_ml_mcp_tools()
    print(f"      ML MCP tools: {[t.name for t in ml_tools]}", flush=True)

    step("build agent (5 local + 2 ML)")
    agent = RequirementAgent(
        llm=llm,
        memory=UnconstrainedMemory(),
        tools=[*local_tools, *ml_tools],
        name="MARGE-Diag",
        description="diagnostic",
        instructions=bundle.system_prompt,
    )

    step("run agent: simple no-tool prompt")
    result = await agent.run("Say HELLO_DIAG and stop. Do not call any tool.")
    print(f"      <- result type: {type(result).__name__}", flush=True)
    answer = getattr(result, "answer", None)
    if answer:
        text = getattr(answer, "text", str(answer))
        print(f"      answer: {text}", flush=True)

    step("trajectory after no-tool run")
    print(f"      {bundle.enforcer.trajectory}", flush=True)

    step("DONE")


if __name__ == "__main__":
    asyncio.run(main())
