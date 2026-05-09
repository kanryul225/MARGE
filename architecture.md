# Architecture

This document captures the architecture, design intent, and folder structure for the clinical-ML orchestration system described in [`overview.md`](./overview.md). It is the reference for _why_ the code is laid out the way it is. When the structure or stack changes, this file changes first.

---

## 1. Technology Choices

| Layer                                          | Choice                                                                                      | Why                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language / runtime                             | **Python (single venv)**                                                                    | XGBoost / CATBoost / SHAP are Python-native; Streamlit is Python; BeeAI Framework + MCP have Python SDKs. A single language keeps the hackathon stack tight.                                                                                                                                                                                                                                                                     |
| Orchestrator agent                             | **BeeAI Framework** (IBM Research, open source — ReAct-style tool-use loop)                 | We want a _dynamic_ loop — the orchestrator decides at runtime which ML tool to call, iterates on results, and re-plans. BeeAI gives this natively (ReAct + "Bee" agent modes). We explicitly rejected a hardcoded LangGraph DAG because it would not scale to "drop in a new ML model and the orchestrator just uses it." BeeAI is also IBM's first-party agent framework, which aligns with the hackathon's IBM-stack scoring. |
| LLM backbone                                   | **Granite 3.x via watsonx.ai** (primary) with Anthropic Claude as a swappable fallback      | Granite is IBM's open-weight LLM family with native tool calling and is the on-stack choice for this hackathon. BeeAI's model adapter is provider-agnostic, so we keep the option to swap to Claude if Granite's tool-use quality blocks us during the build.                                                                                                                                                                    |
| Medical expert agent                           | **BeeAI sub-agent** (separate `Agent` instance, distinct system prompt and retriever scope) | Same framework, scoped tools. Invoked by the orchestrator as a tool call (agent-as-tool pattern).                                                                                                                                                                                                                                                                                                                                |
| ML model exposure                              | **MCP server** (FastMCP)                                                                    | Each ML model is a tool with a self-describing schema (inputs, outputs, training dataset, test accuracy, XAI). The orchestrator discovers tools via MCP rather than via hardcoded imports — adding a model is a file drop. BeeAI has a first-party MCP client.                                                                                                                                                                   |
| Frontend                                       | **Streamlit**                                                                               | Single-stack Python, fast to ship, native widgets for file upload and chart rendering (SHAP plots, prediction cards).                                                                                                                                                                                                                                                                                                            |
| Schemas                                        | **Pydantic v2**                                                                             | Shared types across orchestrator, MCP server, expert agent, and UI.                                                                                                                                                                                                                                                                                                                                                              |
| Package / dep mgmt                             | **uv** workspace                                                                            | One lockfile, fast resolution, monorepo-friendly.                                                                                                                                                                                                                                                                                                                                                                                |
| Deployment (optional, for IBM-stack alignment) | **IBM Cloud Code Engine**                                                                   | If we ship a hosted demo, deploying on IBM Cloud strengthens the IBM-stack story. Out of scope for the local-first MVP.                                                                                                                                                                                                                                                                                                          |

### Why BeeAI over LangGraph (recap)

BeeAI keeps the control flow inside the LLM — the orchestrator chooses tools turn by turn, can re-call tools, and reasons over intermediate results. LangGraph keeps control flow in the code: nodes and edges are explicit. For an "ML head researcher" that has to choose models _based on what it sees_, the BeeAI approach matches the metaphor; the graph approach would force us to enumerate every decision branch in advance.

The trade-off — losing graph-level guarantees about flow — is mitigated by enforcing constraints structurally via tool design and BeeAI middleware (see §2).

### Provider abstraction

To preserve the Granite-primary / Claude-fallback option, the LLM client lives behind a thin wrapper in `packages/llm_provider/`. Both `apps/orchestrator/` and `services/medical_expert_agent/` consume that wrapper, never the raw watsonx or Anthropic SDK directly. Switching providers is a one-line config change.

---

## 2. How Overview Constraints Map to Code

The overview imposes hard rules on the orchestrator. Each is enforced by a specific mechanism, not by prompting alone.

| Constraint (from overview.md)                                                   | Enforcement mechanism                                                                                                                                                                                                                          | Code location                                                                                 |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| "Orchestrator SHOULD NOT make an arbitrary medical decision"                    | The orchestrator has _no_ tool that returns medical advice from its own reasoning. The only path to user is `final_report`, which is gated by a BeeAI middleware (pre-tool-call event) that checks ML + expert tools were called in this turn. | `apps/orchestrator/tools/final_report.py`, `apps/orchestrator/middleware/enforce_protocol.py` |
| "Should use niche ML models … and get confirmed by medical domain-expert agent" | Same middleware checks that at least one ML tool result _and_ one `consult_medical_expert` result are present in the trajectory before `final_report` is allowed.                                                                              | `apps/orchestrator/middleware/enforce_protocol.py`                                            |
| "Abstention and asking back to user"                                            | Two dedicated tools — `abstain` and `ask_user_back` — are valid terminal calls _without_ the ML+expert precondition. The orchestrator is prompted to prefer them when confidence is low or models conflict irresolvably.                       | `apps/orchestrator/tools/abstain.py`, `apps/orchestrator/tools/ask_user_back.py`              |
| "Asynchronous predictions supporting"                                           | The MCP server exposes each model as an independent tool; Claude Agent SDK fans out tool calls in parallel. No code change in the orchestrator is needed to parallelize.                                                                       | `services/ml_mcp_server/server.py`                                                            |
| "Main agent loop should be light"                                               | Orchestrator package contains _only_ tool definitions, prompt, hooks, and the agent constructor. All compute lives in `services/`.                                                                                                             | `apps/orchestrator/` (size budget)                                                            |
| "Different retrievers for orchestrator vs medical expert"                       | Two separate services with no shared module. Orchestrator can only see `patient_data_retriever`; expert agent can only see its own knowledge retriever.                                                                                        | `services/patient_data_retriever/`, `services/medical_expert_agent/retrievers/`               |
| "Each ML model has fixed input/output, accuracy, XAI"                           | Every model implements a single `MLModel` ABC that requires `predict_schema`, `output_schema`, `metadata` (dataset, accuracy), and `explain` (SHAP). The MCP registry refuses to register a model that doesn't provide all four.               | `services/ml_mcp_server/models/_base.py`, `services/ml_mcp_server/registry.py`                |
| "ML models can be added flexibly"                                               | Adding a model = one file in `services/ml_mcp_server/models/`. Registry auto-discovers it; orchestrator picks it up via MCP without code change.                                                                                               | `services/ml_mcp_server/registry.py`                                                          |
| "Medical claims must be sourced" (operationalising the reliability mandate)     | The medical expert returns `MedicalExpertResponse(reasoning, citations)`. A BeeAI middleware rejects responses that contain clinical claims with an empty `citations` list, forcing a re-answer that pulls from a retrieval tool first.        | `services/medical_expert_agent/middleware/enforce_citation.py`, `packages/schemas/retrieval.py` |

---

## 3. Retrieval & Search Layer

The system has two distinct retrieval surfaces, scoped by who uses them. They are kept structurally separate so the orchestrator never sees medical knowledge directly, and the expert never sees patient records directly.

### 3.1 Patient data (orchestrator-side)

Both seeded sample patients and user-uploaded CSV resolve to the same `PatientRecord` schema, so downstream code never branches on source.

- **Seeded SQLite DB** — a small set of curated patients with realistic longitudinal data, addressed by IDs like `seed-042`. Drives narrative-style demos ("analyse patient #42").
- **CSV upload adapter** — in-memory wrapper for files uploaded via Streamlit, addressed by an upload handle the UI issues.

Exposed as a single MCP server, `patient_data_mcp_server`, with one tool: `get_patient_record(handle: str) → PatientRecord`. Using MCP here keeps the orchestrator's tool surface uniform with the ML tools (everything the orchestrator calls is MCP-shaped, except its three local control tools — `consult_expert`, `ask_user_back`, `abstain`, `final_report`).

### 3.2 Medical knowledge (expert-side)

The medical expert agent picks per query between two **independent search tools** that return the same `RetrievedDocument` shape:

- `search_local_medical_kb(query) → list[RetrievedDocument]` — RAG over a curated, embedded corpus (WHO / CDC / MedlinePlus guidelines and similar). Fast, consistent, citable. Limited coverage.
- `search_medical_web(query) → list[RetrievedDocument]` — live web search via Tavily or Exa. Wide coverage, current. Less consistent, requires API key.

Keeping them as **separate tools** (not a single hybrid wrapper) lets the expert pick deliberately and lets each result carry an explicit `retrieval_source` flag — useful for citation provenance and for letting the expert weigh the relative authority of each source.

### 3.3 Deep research subagent (on-demand)

For queries that need iterative research ("which 2024 guidelines disagree on the X cutoff and why"), the expert can spawn a `deep_medical_research` sub-subagent as a tool call. The subagent runs its own BeeAI loop — calling `search_local_medical_kb` and `search_medical_web` repeatedly, refining queries, judging relevance — and returns a synthesized `ResearchReport(findings, citations)`.

This is **on-demand, not always**. The expert's default path is one or two single-shot search calls; the subagent only spawns when the expert explicitly judges the question needs multi-step research. This keeps cost and latency bounded for common queries and reserves the heavier path for cases that justify it.

### 3.4 Citation enforcement

The medical expert's response is shaped as `MedicalExpertResponse(reasoning: str, citations: list[Citation])`. A BeeAI middleware on the expert agent — `enforce_citation` — inspects the response before it returns to the orchestrator: if `reasoning` makes any clinical claim and `citations` is empty, the response is rejected and the expert is told to re-answer with sources. Citations carry source URL, snippet, retrieval timestamp, and the `retrieval_source` flag.

This makes "no medical claim without a source" a structural property, not a prompt habit — and it propagates: the orchestrator's `final_report` can quote citations from the expert response directly into the user-facing output.

---

## 4. Folder Structure

```
IBMHackathon/
├── pyproject.toml                 # uv workspace root
├── README.md
├── overview.md                    # product/concept (existing)
├── architecture.md                # this file
│
├── apps/
│   ├── streamlit_ui/              # User-facing chat interface
│   │   ├── app.py                 # entry: chat + patient-data upload
│   │   ├── components/            # SHAP plot, prediction card, expert-quote card
│   │   └── state.py               # st.session_state wrapper
│   │
│   └── orchestrator/              # The "ML head researcher" agent
│       ├── agent.py               # BeeAI Agent assembly: provider, prompt, tools, middleware
│       ├── system_prompt.md       # role + hard rules; loaded by agent.py
│       ├── tools/                 # *Local* tools only — ML tools live in MCP
│       │   ├── consult_expert.py  # invokes medical_expert_agent
│       │   ├── patient_history.py # calls patient_data_retriever service
│       │   ├── ask_user_back.py
│       │   ├── abstain.py
│       │   └── final_report.py    # ONLY path to user; gated by middleware
│       └── middleware/
│           └── enforce_protocol.py # BeeAI pre-tool event: blocks final_report unless ML+expert called
│
├── services/                      # Heavy work lives here, never in the agent loop
│   ├── ml_mcp_server/             # MCP server exposing every ML model as a tool
│   │   ├── server.py              # FastMCP entry; one tool per registered model
│   │   ├── registry.py            # auto-discovers files in models/ at startup
│   │   ├── models/                # one file per model — drop-in extension point
│   │   │   ├── _base.py           # MLModel ABC (metadata, schema, predict, sample_inputs)
│   │   │   ├── breast_cancer_xgb.py    # XGBoost on Wisconsin Diagnostic
│   │   │   ├── diabetes_catboost.py    # CatBoost on Pima Indians (OpenML)
│   │   │   └── ...                # add new models here, no other edits needed
│   │   ├── explainers/
│   │   │   └── shap_wrapper.py    # shared SHAP utility for all models
│   │   └── artifacts/             # serialized .joblib + JSON metadata (gitignored)
│   │
│   ├── medical_expert_agent/      # The "professional doctor" sub-agent
│   │   ├── agent.py               # BeeAI Agent (sub-agent)
│   │   ├── system_prompt.md       # medical-reasoning persona
│   │   ├── middleware/
│   │   │   └── enforce_citation.py # rejects clinical-claim responses with empty citations
│   │   ├── tools/                 # what the expert can call (scoped to medical knowledge)
│   │   │   ├── search_local_kb.py # RAG-backed search tool (uses packages/medical_kb)
│   │   │   ├── search_web.py      # Tavily/Exa-backed search tool
│   │   │   └── deep_research.py   # spawns deep_medical_research sub-agent on demand
│   │   └── subagents/
│   │       └── deep_medical_research.py # iterative research loop, used only when needed
│   │
│   └── patient_data_mcp_server/   # MCP server exposing patient records (orchestrator-only)
│       ├── server.py              # FastMCP entry; tool: get_patient_record(handle)
│       ├── sources/               # one source = one file
│       │   ├── _base.py           # PatientSource ABC (resolve, list)
│       │   ├── sqlite_db.py       # seeded sample-patient SQLite DB
│       │   └── csv_upload.py      # in-memory adapter for Streamlit uploads
│       └── seed/                  # gitignored: seed_patients.sqlite + fixture CSVs
│
├── packages/                      # Shared code, no runtime services
│   ├── schemas/                   # Pydantic models — single source of truth
│   │   ├── prediction.py          # Prediction, XAIScore, ModelMetadata
│   │   ├── patient.py             # PatientRecord, ClinicalFeature
│   │   ├── retrieval.py           # RetrievedDocument, Citation, ResearchReport,
│   │   │                          # MedicalExpertResponse (response schema enforced by middleware)
│   │   └── orchestration.py       # ConsultationLog, ToolCallTrace (audit)
│   │
│   ├── llm_provider/              # Thin wrapper over BeeAI's model adapter
│   │   ├── client.py              # `get_llm()` — returns Granite (watsonx.ai) by default,
│   │   │                          # Anthropic if WATSONX is unavailable or LLM_PROVIDER overrides
│   │   └── settings.py            # env-var contract for providers + external search APIs
│   │
│   ├── medical_kb/                # Local RAG corpus: ingestion + query
│   │   ├── ingest.py              # download → chunk → embed → Chroma persist dir
│   │   ├── query.py               # vector-store client used by search_local_kb tool
│   │   ├── sources.yaml           # curated source list (WHO/CDC/MedlinePlus URLs)
│   │   └── corpus/                # gitignored: downloaded PDFs + Chroma persist dir
│   │
│   ├── ml_training/               # Offline: train + serialize the niche models
│   │   ├── train_diabetes.py
│   │   └── notebooks/
│   │
│   └── eval/                      # Orchestration-quality eval harness
│       ├── scenarios/             # "conflicting predictions", "missing feature",
│       │                          # "low-confidence ensemble", "abstain-required"
│       └── runner.py
│
├── data/                          # gitignored
│   ├── datasets/                  # training data
│   └── sample_patients/           # demo cases for the UI
│
├── tests/
│   ├── unit/                      # per-package pytest
│   ├── integration/               # MCP server <-> orchestrator wiring
│   └── e2e/                       # Streamlit + full stack
│
└── scripts/
    ├── run_dev.sh                 # boot MCP server + Streamlit + expert agent
    └── train_all.sh               # rebuild all ML artifacts
```

### Layering rules

1. **`apps/` depends on `services/` and `packages/`. Never the reverse.**
2. **`services/` depend only on `packages/`.** Services are independent — the MCP server cannot import from the expert agent, and vice versa.
3. **`packages/schemas/` is the only module imported by all of the above.**
4. **`apps/orchestrator/` cannot import from `services/medical_expert_agent/` internals** — the orchestrator only sees the agent through the `consult_expert` tool. The orchestrator therefore never gains direct access to medical KB or web search tools.
5. **`services/medical_expert_agent/` cannot import from `services/patient_data_mcp_server/`** — the expert never reads patient records directly; if it needs them, the orchestrator includes the relevant fields in the consultation payload.
6. **`packages/medical_kb/` is imported only by `services/medical_expert_agent/tools/search_local_kb.py`** — keeps the vector-store dependency contained to the agent that actually uses it.

---

## 5. Runtime Data Flow

```
Streamlit UI
   │ user uploads CSV (or selects seeded patient #42) + asks "analyse this patient"
   ▼
Orchestrator agent loop (BeeAI, model = Granite via watsonx.ai)
   │ turn 1: get_patient_record(handle=…)              ──MCP──▶ patient_data_mcp_server
   │ turn 2: predict_diabetes_xgb(features=…)          ──MCP──▶ ml_mcp_server
   │ turn 2: predict_cvd_catboost(features=…)          ──MCP──▶ ml_mcp_server   (parallel)
   │ turn 3: consult_medical_expert(findings=…)
   │              │ medical_expert_agent loop (BeeAI sub-agent)
   │              │   ├─ search_local_medical_kb(…)   (default path)
   │              │   ├─ search_medical_web(…)        (when KB insufficient)
   │              │   └─ deep_medical_research(…)     (on-demand sub-sub-agent)
   │              ▼ middleware: enforce_citation passes
   │              ▼ returns MedicalExpertResponse(reasoning, citations)
   │ turn 4: [middleware: enforce_protocol passes — ML✓, expert✓]
   │         final_report(…)                            ──▶ Streamlit
   ▼
Streamlit renders: prediction card + SHAP plot + expert quote with citations + confidence note
```

If turn 3's expert response contradicts ML or confidence stays low, the orchestrator instead calls `ask_user_back` (request a missing feature) or `abstain` (recommend a real doctor) — both terminate the turn without the middleware ever seeing `final_report`.

---

## 6. Extension Points

These are the places we add functionality. Each is a single, well-defined edit.

| To add…                                  | Edit                                                                                                                       | Notes                                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| A new ML model                           | One new file in `services/ml_mcp_server/models/` implementing the `MLModel` ABC.                                           | Registry auto-discovers. Orchestrator gains the tool on next MCP refresh.                                          |
| A new patient data source                | One new file in `services/patient_data_mcp_server/sources/` implementing `PatientSource`.                                  | Auto-registered. The `get_patient_record` tool dispatches by handle prefix.                                        |
| A new medical search tool (extra source) | One new module in `services/medical_expert_agent/tools/` returning `list[RetrievedDocument]`.                              | Expert agent gains it on next restart. Citation flow is automatic via the shared schema.                           |
| A new web search provider                | Swap implementation inside `services/medical_expert_agent/tools/search_web.py`.                                            | Settings live in `packages/llm_provider/settings.py` (unified env-var contract for external APIs).                 |
| Add a source to the local medical KB     | Add URL/path to `packages/medical_kb/sources.yaml`, run `python -m packages.medical_kb.ingest`.                            | No code change — corpus and Chroma index are rebuilt.                                                              |
| A new abstention or ask-back variant     | New tool in `apps/orchestrator/tools/`.                                                                                    | Update `enforce_protocol.py` if it should bypass the ML+expert precondition.                                       |
| A new eval scenario                      | New file in `packages/eval/scenarios/`.                                                                                    | Used by `runner.py` to score orchestrator behavior.                                                                |

---

## 7. Open Questions

To revisit before/during implementation:

- **Granite version** — confirm the exact Granite 3.x variant on watsonx.ai with the strongest tool-calling quality (likely `granite-3-8b-instruct` or the latest available). Validate with a minimal tool-call test before committing the orchestrator to it.
- **watsonx.ai access** — confirm hackathon credits / API key path. Decide between (a) watsonx.ai SaaS, (b) self-hosted Granite via Hugging Face / Ollama as a backup if watsonx access is gated.
- **Embedding model for the local medical KB** — pick from sentence-transformers (`bge-small-en-v1.5`, broadly used and tiny), a Granite embedding if available on watsonx.ai, or an OpenAI-style remote embedder. Trade-off: local models keep ingestion offline and free; remote models give better recall.
- **Web search provider** — Tavily vs Exa vs Brave. Decide by free-tier rate limits, citation richness (snippet + URL + title), and how clean the medical-domain results are. Tavily is the current default in the doc but not committed.
- **Local KB corpus scope** — exact list of sources to seed `sources.yaml`. Must balance breadth (covering plausible demo questions) against ingestion time. WHO / CDC / MedlinePlus is the proposed starting set; add NICE / KDIGO / ADA selectively if cheap to obtain.
- **What counts as a "clinical claim" for citation enforcement** — the middleware must classify expert reasoning into "needs sourcing" vs "general framing." Heuristic options: (a) any sentence that names a number, drug, threshold, or guideline; (b) LLM self-classification; (c) require citations always when `reasoning` is non-empty. Start with (c) and relax if it over-blocks.
- **Patient data format** — minimum viable: CSV upload mapped to a `PatientRecord` schema. FHIR support is out of scope for the hackathon.
- **Audit trail surfacing** — whether to expose the orchestrator's tool-call trace + the expert's citations in the Streamlit UI as a "reasoning log" for transparency. Likely yes — it's a strong demo signal and aligns with the IBM "trustworthy AI" framing.
- **Authentication / multi-user** — out of scope for hackathon.
- **IBM Cloud deployment** — local-only is fine for the MVP, but a deployed Code Engine demo strengthens the IBM-stack story for judging. Decide late based on remaining time.
