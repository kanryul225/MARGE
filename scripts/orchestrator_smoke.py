"""Manual end-to-end smoke for the orchestrator with a live LLM.

This is the only path that actually calls a real LLM, so it's not part of
the unit/integration test suite. Run it once to verify the full happy path:
patient -> ML tools -> medical expert (stub) -> final_report.

Setup:
    export ANTHROPIC_API_KEY=sk-...
    # or for IBM stack:
    # export LLM_PROVIDER=watsonx
    # export WATSONX_API_KEY=...
    # export WATSONX_PROJECT_ID=...
    # export WATSONX_URL=https://us-south.ml.cloud.ibm.com

Run:
    uv run python scripts/orchestrator_smoke.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.orchestrator.agent import build_bundle, build_orchestrator_agent
from packages.llm_provider.client import build_chat_model
from packages.llm_provider.settings import LLMSettings


_PROMPT = (
    "Analyse seed patient `seed-001`. "
    "Use the available ML tools, then consult the medical expert. "
    "Produce a final report."
)


async def main() -> None:
    print("=" * 64)
    print(" MARGE orchestrator end-to-end smoke (LIVE LLM)")
    print("=" * 64)

    settings = LLMSettings.from_env()
    print(f"\nProvider: {settings.provider.value} ({settings.model_id})")

    llm = build_chat_model(settings)
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=llm)

    print("\nRunning agent...\n")
    result = await agent.run(_PROMPT)

    print("\n" + "=" * 64)
    print(" Result")
    print("=" * 64)
    print(result.answer.text if hasattr(result, "answer") else result)

    print("\n" + "=" * 64)
    print(" Trajectory")
    print("=" * 64)
    for i, step in enumerate(bundle.enforcer.trajectory, 1):
        print(f"  {i:2d}. {step}")

    final_called = bundle.enforcer.has_called("final_report")
    print(f"\n[ok] final_report reached: {final_called}")
    if not final_called:
        print("[note] orchestrator did not produce final_report — check trajectory above")


if __name__ == "__main__":
    asyncio.run(main())
