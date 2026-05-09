# Project notes

Operational notes that don't belong in `architecture.md` or `README.md`.

## Live-LLM verification status (May 2026)

What works (verified):

- All 5 providers register and `build_chat_model()` returns the right adapter.
- Direct `ChatModel.run([UserMessage(...)])` succeeds against:
  - **NVIDIA NIM** (`qwen/qwen3.5-397b-a17b`)
  - **Cerebras** (`qwen-3-235b-a22b-instruct-2507`)
- A single agent iteration (`agent.requirement.start → success`) completes:
  - Cerebras: ~1.1s per iteration
  - NVIDIA NIM: ~8.8s per iteration
- The 7-tool surface (5 local + 2 ML MCP) registers correctly on the agent.

What doesn't yet work (deferred):

- **Multi-iteration agent loop** against free-tier providers stalls. Two
  distinct failure modes observed:
  - **Cerebras**: 429 `request_quota_exceeded` after the second LLM call.
    The 30 RPM limit is enforced strictly; back-to-back calls in <2s burst
    hit it.
  - **NVIDIA NIM**: the second iteration's request hangs indefinitely
    (no response after 5+ minutes, only ~5s of CPU). Direct multi-call
    sequences and direct tool-call requests both work fine, so the hang
    is likely specific to how BeeAI sends the tool-result follow-up
    (or a per-key concurrency cap on NIM serverless).
- **Chutes**: marketed as free, but the API returns
  `402 Quota exceeded and account balance is $0.0`. Configured but unusable
  until the account has TAO/credits.

What to add next slice:

1. **Per-provider rate limiter** in `packages/llm_provider/`: a small
   wrapper that enforces a min-gap between calls (Cerebras → 2.2s,
   NVIDIA → 1.6s). Should subclass or duck-type ChatModel and intercept
   `run()` to insert `asyncio.sleep` as needed.
2. **Retry-on-429 with exponential backoff**: BeeAI has `Retryable`
   utilities — wire them into the LLM adapter or agent so transient
   429s self-recover.
3. **Switch test path to Anthropic**: for development the hackathon
   doesn't need the IBM-stack endpoint — Anthropic Haiku in the dev
   loop, swap to NVIDIA/Cerebras only for the demo recording.
4. **Investigate the NVIDIA hang**: capture the exact HTTP request
   (headers + body) BeeAI sends on the second iteration, replay with
   `curl` to isolate whether it's a NIM-side issue or a BeeAI
   serialisation bug.

## Verification scripts

- `scripts/smoke_test.py` — ML model + MCP smoke (all green, in CI).
- `scripts/diag_agent.py` — step-by-step agent build (no tool use).
  Always passes — confirms single-iteration agent works.
- `scripts/diag_agent_tooluse.py` — tool-using agent + BeeAI emitter
  event tracing. Run unbuffered:
  ```bash
  uv run python -u scripts/diag_agent_tooluse.py > /tmp/marge.log 2>&1
  tail -F /tmp/marge.log
  ```
- `scripts/orchestrator_smoke.py` — full live demo path. Currently
  hits the multi-iteration issues above; will work once the rate
  limiter and retry layer land.

## Observation: BeeAI deprecations to address

- `ToolCallingAgent` is deprecated → migrated to `RequirementAgent` already.
- `agent.emitter.match("*.*", ...)` is deprecated → switch to `on(...)`
  in `diag_agent_tooluse.py`.
- `RequirementAgent`'s `Requirement` system could express the
  protocol invariants (ML-then-expert-then-final_report) declaratively
  and replace `enforce_protocol` middleware in a future refactor.
