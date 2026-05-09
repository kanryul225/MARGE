"""Tool-using diagnostic: subscribe to BeeAI events and print each step.

Reveals exactly which tool the agent calls (or fails to call) and how long
each LLM round trip takes. Bypasses the orchestrator_smoke.py pipe so output
is unbuffered.

Run: `uv run python -u scripts/diag_agent_tooluse.py`
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


async def main() -> None:
    from apps.orchestrator.agent import build_bundle, orchestrator_agent
    from packages.llm_provider.client import build_chat_model_for_role
    from packages.llm_provider.settings import Role

    print("[diag] building llm + agent...", flush=True)
    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle()

    t0 = time.time()

    def stamp() -> str:
        return f"+{time.time() - t0:5.1f}s"

    prompt = (
        "You're given seed patient `seed-001`. "
        "Step 1: get_patient_history. "
        "Step 2: predict_diabetes_risk. "
        "Step 3: consult_medical_expert with a one-line summary. "
        "Step 4: final_report."
    )

    async with orchestrator_agent(bundle=bundle, llm=llm) as agent:
        print(f"[diag] llm: {llm.model_id}, tools: {len(agent._tools)}", flush=True)

        def on_any(data, event):
            name = getattr(event, "name", "?")
            path = getattr(event, "path", "")
            if name in {"new_token", "partial_update"}:
                return
            print(f"[event {stamp()}] {path or name}", flush=True)

        agent.emitter.match("*.*", on_any)

        print(f"\n[diag {stamp()}] running agent...\n", flush=True)
        try:
            async def _run():
                return await agent.run(prompt)
            result = await asyncio.wait_for(_run(), timeout=180)
        except asyncio.TimeoutError:
            print(f"\n[diag {stamp()}] TIMEOUT after 180s", flush=True)
            print(f"[diag] trajectory so far: {bundle.enforcer.trajectory}", flush=True)
            return

        print(f"\n[diag {stamp()}] DONE. result type={type(result).__name__}", flush=True)
        answer = getattr(result, "answer", None)
        if answer:
            print(f"[diag] answer: {getattr(answer, 'text', str(answer))[:500]}", flush=True)

        print(f"[diag] trajectory: {bundle.enforcer.trajectory}", flush=True)
        print(f"[diag] final_report reached: {bundle.enforcer.has_called('final_report')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
