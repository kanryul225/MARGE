# MARGE

Multi-agent ML-Reasoning Guidance Engine — a clinical diagnosis assist system that wraps niche tabular ML models behind a dynamic LLM orchestrator, with a separate medical expert sub-agent for clinical reasoning.

See [`overview.md`](./overview.md) for the concept and [`architecture.md`](./architecture.md) for design intent + folder structure.

## Status

This repository is at the **thin-slice** stage: ML model + MCP server work end-to-end. Orchestrator (BeeAI), medical expert sub-agent, and Streamlit UI are skeletoned but not yet wired up.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Create the venv and install core dependencies
uv sync

# 2. Train the demo ML models (writes joblib artifacts under services/ml_mcp_server/artifacts/)
uv run python -m packages.ml_training.train_breast_cancer
uv run python -m packages.ml_training.train_diabetes

# 3. Smoke test: walks every registered model and verifies direct call == MCP call
uv run python scripts/smoke_test.py
```

Optional dependency groups (install when working on those layers):

```bash
uv sync --extra orchestrator   # BeeAI Framework + watsonx.ai + anthropic
uv sync --extra ui             # Streamlit + matplotlib + plotly
uv sync --extra medical-kb     # Chroma + sentence-transformers + Tavily
uv sync --extra dev            # pytest + ruff
```

## Layout

```
apps/         user-facing entrypoints (orchestrator, streamlit_ui)
services/     long-running components (MCP servers, sub-agents)
packages/     shared libraries (schemas, llm_provider, medical_kb, eval)
scripts/      operational scripts (smoke test, training, dev runner)
tests/        unit / integration / e2e
```

The orchestrator (`apps/orchestrator/`) only depends on `services/` and `packages/`. Services do not depend on each other. See `architecture.md` §4 for the full layering rules.
