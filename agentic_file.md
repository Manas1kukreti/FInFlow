# FinFlow — Comprehensive Architecture Document

---

## 1. Executive Summary & Context

FinFlow is a finance workflow console. Users upload source files, write a natural language instruction, choose an output format, and receive a downloadable processed artifact. The system routes every job through a structured multi-agent AI pipeline that interprets the instruction, cleans and normalizes the data, filters it, runs calculations, visualizes results, and assembles the final output.

The agent pipeline is the core of FinFlow. Everything else — the React frontend, the FastAPI backend, PostgreSQL, Redis — exists to support the pipeline and surface its results to users.

**Design Philosophy & Constraints:**

- Agent cap: **10 to 15 agents maximum**. This ceiling is deliberate. It avoids over-engineering for scale that is not needed. Every agent addition requires a deliberate code change and a PR review.
- No `AgentExecutor`. No `initialize_agent`. LangGraph only. See Section 9.
- No semantic retrieval for agent selection. The Orchestrator dumps all registered agents via `describe_all()` and builds the DAG from the full list. Static decorator-based registration at import time is sufficient.
- The Orchestrator generates the full execution plan (DAG) upfront. A deterministic engine dispatches and walks it. The LLM plans once; the engine executes deterministically.
- Debugging must be traceable. Each agent has one job. If output is wrong, the failure is isolated to exactly one agent.

---

## 2. Full System Architecture

FinFlow has three parts that work together:

```
Browser
  React + Vite SPA
    Auth pages
    Upload Center
    Submissions / Dashboard / Manager / Admin / Audit / Settings
        |
        v
FastAPI Backend  (port 8000)
  Auth
  Uploads
  Downloads
  Alerts
  Analytics
  Admin
  Agent Callback
        |
        +--> PostgreSQL (port 5433)
        +--> Redis dispatch queue (port 6379)
        +--> Agent Service (port 8001)
                |
                v
        7-Agent DAG Pipeline
          Ingestion Agent
          Data Cleaning & Normalization Agent
          Filter Agent
          Calculation Agent
          Visualization Agent
          Reporting Agent
          (Orchestrator governs all)
                |
                v
        Output artifact written to disk
                |
                v
        Backend callback stores output path
                |
                v
        Frontend download endpoint serves file
```

---

## 3. Core Workflow

1. A user uploads a file and writes a natural language instruction, selecting an output format.
2. The backend stores a `submissions` row and marks it `pending`.
3. The backend enqueues the submission id into Redis.
4. The agent service consumes the queue item.
5. The Orchestrator classifies the intent, validates the domain, builds an `ExecutionPlan` (DAG), and dispatches it to the execution engine.
6. The engine walks the DAG, running each agent step in topological order.
7. The Reporting Agent writes the final output artifact to disk.
8. The agent service posts the result to `POST /api/agent/callback`.
9. The backend stores the output file path, updates submission status, and emits live refresh events via WebSocket.
10. The frontend exposes the download action when the job is `complete`.

---

## 4. Status Model

There are two separate status layers. They must never be collapsed into one.

### 4.1 Submission-Level Status (User-Facing, Backend)

This is what the UI shows and what the backend stores on the `submissions` table.

| Status | Meaning |
|---|---|
| `pending` | Job created, not yet picked up by the agent service |
| `processing` | Agent DAG is actively running |
| `complete` | All steps succeeded, output artifact is available |
| `failed` | One or more steps failed, no usable artifact |
| `quarantined` | Job flagged for human review — unsupported domain or capability |

### 4.2 Step-Level Status (Internal, DAG Engine)

This is what each `AgentResult` envelope returns. It lives in memory during execution and is written to `audit_logs` on completion. It never goes directly to the `submissions` table.

| Status | Meaning |
|---|---|
| `success` | Step completed, output is in `PipelineState.data` |
| `partial` | Step completed but with warnings or incomplete output |
| `failed` | Step failed with an error message |

### 4.3 Status Mapping Contract

The engine applies this mapping to determine the final submission status after DAG execution:

- All steps return `success` → submission becomes `complete`
- Any step returns `failed` → submission becomes `failed` immediately (fail loudly)
- Any step returns `partial` → submission becomes `failed` (partial is not silently accepted)
- Unsupported domain or capability detected by Orchestrator → submission becomes `quarantined`

`partial` maps to `failed` by design. Partial output in a finance pipeline is more dangerous than no output. Fail loudly so a human reviews it.

---

## 5. The 7-Agent Pipeline

The pipeline is a Directed Acyclic Graph (DAG). A shared `PipelineState` object is passed between agent nodes. The Orchestrator builds the DAG upfront. The execution engine dispatches it deterministically.

### Agent Count

| # | Agent | Stage |
|---|-------|-------|
| 1 | Orchestrator | — (planner, not a DAG node) |
| 2 | Ingestion Agent | `ingest` |
| 3 | Data Cleaning & Normalization Agent | `transform` |
| 4 | Filter Agent | `transform` |
| 5 | Calculation Agent | `analyze` |
| 6 | Visualization Agent | `visualize` |
| 7 | Reporting Agent | `deliver` |

Total: 7. Cap: 15. Headroom: 8 agents remaining.

---

### 5.1 Orchestrator (The Brain)

The Orchestrator is the central planner. It uses an LLM with structured output to understand the user's instruction and build the DAG. It is not a DAG node itself — it runs before the engine and produces the `ExecutionPlan`.

**Input:** User's natural language instruction and uploaded file reference.

**Responsibilities:**
1. Intent classification — determine what operations are needed.
2. Parameter extraction — extract filters, column selections, calculation types, chart preferences, output format.
3. Domain validation — hard-reject unsupported domains before the DAG runs. Unsupported domains go to `quarantined` immediately.
4. Registry capability matching — verify that the requested operations can be handled by registered agents.
5. DAG construction — produce an `ExecutionPlan` consisting of `PlanStep` objects with `depends_on` links.

**Output:** A validated `ExecutionPlan` object. This is the single source of truth for routing. No downstream agent makes routing decisions.

**Rule:** The Orchestrator's prompt dumps all registered agents via `registry.describe_all()`. It selects from this list only. It cannot hallucinate agent names that do not exist in the registry.

---

### 5.2 Ingestion Agent

The Ingestion Agent is the first node in every DAG. Its sole job is to convert the uploaded file into a clean, structured dataframe or text object that the Data Cleaning Agent can consume. It is the only agent that knows about file formats.

**Stage:** `ingest`

**Input:** Raw file reference (path + file type).

**Responsibilities:**
- Tabular files (`.xlsx`, `.xls`, `.csv`, `.tsv`, `.json`) → parsed into a dataframe directly.
- `.txt` files → parsed into structured text.
- `.pdf` files → text extraction into structured text or table.
- Image files (`.png`, `.jpg`, `.jpeg`, `.webp`) → **rejected at this layer** with a clear error. No silent OCR hallucination. The job is quarantined.

**Output:** A structured dataframe or text object keyed into `PipelineState.data` by this step's `step_id`.

**Rule:** If the file type cannot be parsed into a usable structure, the Ingestion Agent returns `AgentResult(status="failed")` immediately. The engine stops and the submission becomes `failed`. It never passes unreadable input downstream.

---

### 5.3 Data Cleaning & Normalization Agent

The Data Cleaning & Normalization Agent sanitizes and standardizes the ingested data so that every downstream agent can make hard assumptions about the data shape. It does two things: structural cleaning and format normalization. These are inseparable — a dataset with clean structure but inconsistent formats is still not safe for downstream processing.

**Stage:** `transform`

**Input:** Structured dataframe from the Ingestion Agent.

**Responsibilities:**

*Structural Cleaning:*
- Type coercion — ensure columns have consistent types
- Null handling — flag, fill, or drop nulls per column type
- Deduplication — remove exact duplicate rows
- Whitespace trimming — strip leading and trailing whitespace from string fields

*Format Normalization:*
- Dates → ISO 8601 (`YYYY-MM-DD`). Every date field, regardless of input format
- Currency → standard decimal with consistent precision (e.g., `1234.56`). No symbols, no commas
- Numbers → consistent decimal precision throughout the dataset
- Categorical strings → consistent casing (e.g., all merchant names lowercase, all status values uppercase)

**Why normalization lives here:** Every downstream agent — Filter, Calculation, Visualization, Reporting — can now make hard assumptions. The Filter Agent knows dates are always `2024-01-15`, not `15/Jan/24`. The Calculation Agent never needs to coerce types before doing math. If a downstream agent produces wrong output due to a format issue, the root cause is always traceable to this single agent. Fix it here; fixed everywhere.

**Output:** Cleaned and normalized dataframe with a detailed cleaning log (columns changed, nulls handled, format conversions applied), keyed into `PipelineState.data` by this step's `step_id`.

---

### 5.4 Filter Agent

The Filter Agent applies row-level and column-level selections to the cleaned, normalized dataframe based on conditions extracted by the Orchestrator from the user's instruction. It has no opinion about data quality — that is already handled by the Cleaning Agent.

**Stage:** `transform` (after Cleaning, enforced by the stage ordering rule)

**Input:** Cleaned and normalized dataframe from the Cleaning Agent.

**Responsibilities:**
- Row filtering — apply WHERE-style conditions (e.g., `merchant = "PayPal"`, `payment_mode = "cash"`, `amount > 5000`, date ranges)
- Column projection — return only the fields specified by the user's instruction, drop the rest
- Multi-condition logic — AND/OR combinations parsed from the user's prompt by the Orchestrator
- If no filter or projection criteria exist in the instruction, the Filter Agent is **not included in the DAG** by the Orchestrator. Zero overhead.

**Why this is a separate agent from Cleaning:** Mixing filtering logic into the Cleaning Agent conflates two different concerns. When output is wrong, you cannot tell whether the Cleaning Agent changed a value incorrectly or the Filter Agent dropped a row incorrectly. Separate agents mean traceable failures. Each agent has one job.

**Output:** Filtered and projected dataframe keyed into `PipelineState.data` by this step's `step_id`.

---

### 5.5 Calculation Agent

The Calculation Agent is the mathematical engine of the pipeline. It operates on clean, normalized, optionally filtered data.

**Stage:** `analyze`

**Input:** Dataframe from the Filter Agent or the Cleaning Agent (whichever is the last `transform` step in the DAG).

**Responsibilities:**
- Perform specific calculations extracted by the Orchestrator: variance, mean, sum, yield, percentage change, running totals, group aggregations.
- Produce a structured result table or series.

**Rule:** The Calculation Agent does not make routing decisions. It does not set `is_final_answer` or any similar flag. Routing is determined entirely by the DAG shape built by the Orchestrator.

**Output:** Structured result table or series keyed into `PipelineState.data` by this step's `step_id`.

---

### 5.6 Visualization Agent

The Visualization Agent creates chart objects for embedding into the final output artifact.

**Stage:** `visualize`

**Input:** Dataframe from any upstream `transform` step, or Calculation Agent output.

**Responsibilities:**
- Select the appropriate chart type based on the Orchestrator's extracted parameters or the data shape.
- Produce native chart objects ready for embedding in XLSX or PDF.
- PNG is supported as an **internal intermediate format only** for embedding purposes. PNG is not a standalone user-facing output format.

**Chart library:** `openpyxl` is the default. `xlsxwriter` is the designated fallback for finance-specific chart types that `openpyxl` cannot render. The fallback is a library flag, not an architectural change.

**Rule:** The Visualization Agent is included in the DAG only when the user's instruction requests a chart or visualization. If not requested, the DAG routes directly from Calculation to Reporting.

**Output:** Chart object(s) keyed into `PipelineState.data` by this step's `step_id`.

---

### 5.7 Reporting Agent

The Reporting Agent is the final assembly point. It takes accumulated outputs from all upstream agents and produces the downloadable artifact in the format the user requested.

**Stage:** `deliver`

**Input:** All upstream step outputs available in `PipelineState.data`.

**Responsibilities:**
- Assemble the final deliverable in the requested output format.
- Embed native chart objects from the Visualization Agent if present.
- Apply display-layer formatting: currency symbols, date display formats, number formatting, column widths. This is the only agent that applies visual/display formatting. The Cleaning Agent normalizes to raw standard form; the Reporting Agent formats for human readability.
- Write a summary section or text description if requested.
- Write the output file to the `OUTPUT_DIR`.

**Supported output formats:**

| Format | Notes |
|---|---|
| `XLSX` | Native Excel with embedded charts if Visualization step was included |
| `CSV` | Flat tabular output |
| `JSON` | Structured JSON |
| `TXT` | Plain text |
| `PDF` | PDF with embedded charts if Visualization step was included |

**PNG is not a supported output format.** PNG is internal to the Visualization Agent only.

If an unsupported format is requested, the job fails clearly. No silent defaulting to Excel.

**Output:** Final output file path, written to disk. This path is what the backend callback stores.

---

## 6. DAG Architecture

### 6.1 Agent Specification Contract

Every agent registers itself via a decorator at import time.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

class AgentSpec(BaseModel):
    name: str = Field(description="Unique identifier for the agent")
    description: str = Field(description="Plain text description for the LLM Orchestrator")
    stage: Literal["ingest", "transform", "analyze", "visualize", "deliver"]
    accepts: List[str] = Field(description="Required input data types (e.g., ['dataframe'])")
    produces: List[str] = Field(description="Output data types (e.g., ['filtered_dataframe'])")
    params_schema: Dict[str, Any] = Field(description="JSON Schema for expected parameters")
```

Registration:

```python
@registry.register
class IngestionAgent:
    spec = AgentSpec(
        name="ingestion_agent",
        description="Parses any supported file format into a structured dataframe or text object.",
        stage="ingest",
        accepts=["file_reference"],
        produces=["dataframe"],
        params_schema={"file_path": {"type": "string"}, "file_type": {"type": "string"}}
    )
```

### 6.2 Execution Plan

The Orchestrator outputs a graph of steps using instance-based IDs. Agent names must match registered `AgentSpec.name` values exactly.

```python
class PlanStep(BaseModel):
    step_id: str          # Unique instance ID, e.g. "step_1", "step_2"
    agent: str            # Must match a registered AgentSpec.name
    params: Dict[str, Any]
    depends_on: List[str] # Upstream step_ids this step relies upon

class ExecutionPlan(BaseModel):
    steps: List[PlanStep]
```

Example plan for "clean this data and return only PayPal merchant rows, then give me a sum of amounts":

```json
{
  "steps": [
    {"step_id": "step_1", "agent": "ingestion_agent", "params": {"file_path": "...", "file_type": "csv"}, "depends_on": []},
    {"step_id": "step_2", "agent": "cleaning_agent", "params": {}, "depends_on": ["step_1"]},
    {"step_id": "step_3", "agent": "filter_agent", "params": {"filters": [{"column": "merchant", "op": "eq", "value": "PayPal"}]}, "depends_on": ["step_2"]},
    {"step_id": "step_4", "agent": "calculation_agent", "params": {"operations": [{"type": "sum", "column": "amount"}]}, "depends_on": ["step_3"]},
    {"step_id": "step_5", "agent": "reporting_agent", "params": {"output_format": "xlsx"}, "depends_on": ["step_4"]}
  ]
}
```

### 6.3 Pipeline State

The shared state bag. Keyed by `step_id`, not agent name. This prevents namespace collisions when the same agent type is used more than once in a DAG.

```python
class PipelineState(BaseModel):
    data: Dict[str, Any] = {}  # e.g., data["step_2"] = cleaned_dataframe
```

### 6.4 Agent Result Envelope

Every agent must return this standard envelope. No exceptions.

```python
class AgentResult(BaseModel):
    status: Literal["success", "partial", "failed"]
    error_message: str | None = None
    data: Any = None
```

### 6.5 Stage Ordering Rule

The `stage` enum enforces monotonic progression. A dependent step must have the same or later stage than its dependency. The engine checks this before executing step 1.

Valid: `ingest → transform → transform → analyze → visualize → deliver`

Invalid (engine rejects): `transform → ingest`, `visualize → analyze`

This structurally prevents nonsensical plans (e.g., filtering before ingestion, visualizing before calculating).

### 6.6 Dynamic Routing

The DAG shape determines routing — not agent flags.

**Full path:**
`Ingestion → Cleaning → Filter → Calculation → Visualization → Reporting`

**Without filter criteria:**
`Ingestion → Cleaning → Calculation → Visualization → Reporting`

**Without visualization:**
`Ingestion → Cleaning → Filter → Calculation → Reporting`

**Minimal path (clean only):**
`Ingestion → Cleaning → Reporting`

A step that has no dependents (nothing else depends on it) and feeds the Reporting Agent is the terminal step.

---

## 7. Execution Engine Rules

The engine runs between the Orchestrator and the first agent step. It validates the plan before executing anything.

1. **Parameter validation** — validate the Orchestrator's `params` for each step against the agent's declared `params_schema`. Reject malformed plans before execution.
2. **Cycle detection** — run a topological sort on the `depends_on` graph. If a cycle exists, reject the plan. The Orchestrator LLM can hallucinate circular dependencies; the engine catches them.
3. **Stage ordering check** — enforce monotonic stage progression across the DAG.
4. **Execution** — dispatch steps in topological order. Steps with no mutual dependencies can run in parallel.
5. **Failure handling** — if any step returns `status: "failed"`, stop the DAG immediately. Do not attempt remaining steps. Update submission to `failed` and write to `audit_logs`.

---

## 8. Persistence Contract

### What lives in memory

`PipelineState.data` is in-memory only during DAG execution. It is never written to the database. Intermediate step outputs (cleaned dataframes, filter results, calculation tables) are transient.

### What gets persisted to PostgreSQL

On DAG completion, the agent service posts to the backend callback with:

```json
{
  "submission_id": "...",
  "status": "complete | failed | quarantined",
  "output_path": "/outputs/filename_processed.xlsx",
  "summary": {
    "steps_run": ["step_1", "step_2", "step_3", "step_4", "step_5"],
    "step_statuses": {"step_1": "success", "step_2": "success", ...},
    "duration_ms": 4200
  }
}
```

The backend stores:
- `output_path` on the `submissions` row
- `status` update on the `submissions` row
- Full `summary` JSON blob on the `submissions` row
- Each step's result written as an append-only entry to `audit_logs`

`submission_records` stores structured records from the final agent output only — not intermediate step results. It stays schema-stable regardless of how many agents are in the pipeline.

### Quarantine and DLQ

Unsupported domains or capabilities detected by the Orchestrator are written to:
- `needs_review_jobs` table in PostgreSQL — jobs requiring human review
- `dead_letter_jobs` table in PostgreSQL — jobs that cannot be retried automatically

The submission status is set to `quarantined`. Admins see this in the Admin Dashboard. Regular employees see a `pending review` state. Nothing is silently buried.

---

## 9. Technology Stack

### Permitted and Mandated

| Component | Choice | Reason |
|---|---|---|
| Orchestration | LangGraph | Built for DAG architectures. Explicit, debuggable, controllable |
| LangChain utilities | `langchain-core`, Prompt Templates, Output Parsers, Document Loaders only | Low-level only |
| Queue | Redis + `arq` | Free, open-source, asyncio-native — critical since agents spend most time waiting on LLM I/O |
| Database | PostgreSQL | Permanent system of record. Job history, step audit trail, DLQ, quarantine all live here |
| API layer | FastAPI | Async-native, pairs with `arq` workers |
| DLQ/Quarantine | Native Postgres tables | No dedicated queue monitoring product. At this scale it is pure overhead |
| Chart default | `openpyxl` | Free, sufficient for most finance charts |
| Chart fallback | `xlsxwriter` | Designated fallback for complex chart types `openpyxl` cannot render |
| LLM backend | Groq | Configured via `GROQ_API_KEY` |

### Strictly Prohibited

`AgentExecutor` and `initialize_agent` from LangChain are banned.

Reasons:
1. Architecture mismatch — `AgentExecutor` forces an opaque step-by-step loop. FinFlow generates a full DAG upfront and executes it deterministically. These are fundamentally incompatible approaches.
2. Debugging — errors inside `AgentExecutor` surface deep in framework internals, not at the decision boundary in the code.
3. LangGraph was built specifically to replace `AgentExecutor` for explicit multi-step orchestration.

**Rule:** Whenever a tutorial or guide suggests `from langchain.agents import AgentExecutor` or `initialize_agent`, skip it and map the concept directly to its LangGraph equivalent (`StateGraph`, nodes, and conditional edges).

Redis is treated as purely disposable. If it restarts and drops the queue, no critical historical data is lost — everything important is in PostgreSQL.

---

## 10. Red-Team Analysis

### 10.1 Untyped Parameters

**Vulnerability:** Untyped `params` allow the Orchestrator LLM to hallucinate parameter keys.

**Fix:** The engine validates the LLM's parameter output against each agent's `params_schema` before execution. Invalid parameters cause plan rejection, not a runtime error mid-DAG.

### 10.2 Namespace Collisions in State

**Vulnerability:** Agents producing flat keys (like `result`) cause overwrites in shared state.

**Fix:** `PipelineState.data` is keyed by `step_id` (e.g., `data["step_2"]`), not agent name. `depends_on` owns topology. `accepts/produces` owns typing. The engine resolves data flow by step instance and validates that a step's produced type matches the dependent step's accepted type.

### 10.3 Nonsensical but Technically Valid Plans

**Vulnerability:** A plan might pass type validation but fail logically (e.g., filtering before cleaning, visualizing before calculating).

**Fix:** Stage-ordering check. The engine enforces monotonic stage progression as a hard structural check. A visualization step cannot depend on an ingestion step directly. The DAG shape must follow `ingest → transform → analyze → visualize → deliver` order.

Regression eval suite: a fixed set of 30–50 test prompts is maintained and re-run against the Orchestrator's plan generation every time an agent is added or modified.

### 10.4 DAG Cycle Hallucinations

**Vulnerability:** The Orchestrator LLM might produce circular `depends_on` arrays.

**Fix:** Cycle detection. The engine runs a topological sort check on the full graph before executing step 1. Circular graphs are rejected immediately.

### 10.5 Undefined Failure Contracts

**Vulnerability:** Agents failing in different ways breaks generic retries and error handling.

**Fix:** Every agent returns a standard `AgentResult` envelope (`status`, `error_message`, `data`). No exceptions. The engine handles all failure routing based on this contract.

### 10.6 Unreadable Input Files

**Vulnerability:** Passing an unreadable file silently downstream produces garbage output.

**Fix:** The Ingestion Agent is the single chokepoint for all file parsing. If it cannot produce a usable structure, it returns `AgentResult(status="failed")` immediately. No downstream agent ever receives unstructured raw bytes.

### 10.7 Partial Output Accepted as Success

**Vulnerability:** A step returning `partial` could silently degrade output quality without surfacing to the user.

**Fix:** `partial` maps to `failed` at the submission level. Finance output must be correct or absent — not partially correct. Users are never shown degraded output as if it were complete.

---

## 11. Agent Service File Structure

```
src/finflow_agent/
  orchestrator.py          # ExecutionPlan, PlanStep builder — LLM planner
  state.py                 # PipelineState only — no legacy fields
  registry.py              # AgentSpec, @registry.register decorator, describe_all()
  engine.py                # Topological sort, cycle detection, stage ordering, step dispatch
  agents/
    ingestion_agent.py     # File parsing — only agent that knows about file formats
    cleaning_agent.py      # Structural cleaning + format normalization
    filter_agent.py        # Row filtering, column projection
    calculation_agent.py   # Math operations
    visualization_agent.py # Chart object generation
    reporting_agent.py     # Final artifact assembly
  tools/
    output.py              # Multi-format output file generation
  api.py                   # FastAPI entrypoint, callback handler
  llm.py                   # LLM helpers, Groq client
```

**Abandoned files (do not recreate):**

The following files from the previous implementation are fully abandoned. None of their logic survives.

- `intent.py`
- Old `registry.py` (capability matching / scoring engine)
- `agents/orchestrator.py`
- `agents/cleaning_agent.py` (old version)
- `agents/ui_agent.py`
- `tools/output.py` (old version)
- `config/agents.yml`

---

## 12. Data Model

### Primary Tables

- `users` — app users, roles, and manager assignments
- `refresh_tokens` — hashed refresh tokens for session rotation
- `submissions` — upload metadata, instruction, output format, agent task id, output path, status, summary JSON, timestamps
- `submission_records` — structured records from final agent output only
- `reviews` — manager decisions for approvals and declines
- `submission_comments` — thread-style comments tied to a submission
- `alerts` — manual or system alerts
- `audit_logs` — immutable append-only audit trail, including per-step results from the DAG
- `registered_agents` — registry of live agent names and statuses

### DLQ and Quarantine Tables

- `needs_review_jobs` — jobs quarantined for human review (unsupported domain or capability)
- `dead_letter_jobs` — jobs that failed and cannot be retried automatically

### Compatibility Table

- `transaction_rows` — exists for legacy compatibility only. The current flow uses `submission_records` and generated output artifacts as the primary data shape. Do not extend this table.

---

## 13. Frontend Structure

```
frontend/src/
  api/
    client.js
    finflow.js
  auth/
    AuthContext.jsx
    ProtectedRoute.jsx
  components/
    PageHero.jsx
    StatCard.jsx
    StatusPill.jsx
    ProgressMilestones.jsx
    CommentThread.jsx
  hooks/
    useLiveJobRefresh.js
  pages/
    AuthPage.jsx
    LandingPage.jsx
    VerifyPasswordPage.jsx
    UploadCenter.jsx
    SubmissionsPage.jsx
    Dashboard.jsx
    ManagerDashboard.jsx
    AdminDashboard.jsx
    AuditPage.jsx
    SettingsPage.jsx
  shell/
    AppShell.jsx
```

The frontend stores the current session in `localStorage` under `ledgerflow_auth` and refreshes tokens through the backend auth endpoints. Password changes use a secure email verification flow that globally revokes all sessions on completion.

---

## 14. Backend Structure

```
backend/app/
  api/
    auth.py
    uploads.py
    comments.py
    approvals.py
    alerts.py
    admin.py
    audit.py
    agent.py
    analytics.py
  core/
    config.py
    security.py
  models.py
  schemas.py
  services/
    agent_dispatcher.py
    excel_parser.py
    submission_results.py
    websocket_manager.py
    audit.py
```

---

## 15. API Reference

### Auth

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login, returns access + refresh tokens |
| POST | `/api/auth/refresh` | Refresh access token |
| POST | `/api/auth/logout` | Logout, revoke refresh token |
| GET | `/api/auth/me` | Get current user profile |
| PATCH | `/api/auth/me` | Update current user profile |
| POST | `/api/auth/change-password` | Send 15-minute email verification link |
| POST | `/api/auth/verify-password-change` | Validate link, globally log out old sessions |

### Uploads

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/uploads` | Create a submission, enqueue to Redis |
| GET | `/api/uploads` | List jobs with filtering |
| GET | `/api/uploads/{upload_id}` | Job preview and version history |
| GET | `/api/uploads/{upload_id}/transactions` | Legacy row output |
| GET | `/api/uploads/{upload_id}/job-detail` | Card-ready job detail payload |
| POST | `/api/uploads/{upload_id}/retry` | Requeue the job |
| GET | `/api/uploads/{upload_id}/download` | Download the generated artifact |

### Comments and Approvals

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/submissions/{submission_id}/comments` | Get comment thread |
| POST | `/api/submissions/{submission_id}/comments` | Post a comment |
| POST | `/api/approvals/approve` | Manager approves a submission |
| POST | `/api/approvals/reject` | Manager rejects a submission |
| GET | `/api/approvals/verify-token` | Verify approval token |

### Alerts and Analytics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/alerts` | List alerts |
| POST | `/api/alerts` | Create an alert |
| PATCH | `/api/alerts/{alert_id}/read` | Mark alert as read |
| PATCH | `/api/alerts/read-all` | Mark all alerts as read |
| GET | `/api/analytics/kpis` | KPI aggregates |

### Admin

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/admin/managers` | List managers |
| GET | `/api/admin/employees` | List employees |
| POST | `/api/admin/assign` | Assign employee to manager |
| POST | `/api/admin/reassign` | Reassign employee |

### Agent

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/agent/login` | Agent service authenticates |
| POST | `/api/agent/upload` | Agent receives job payload |
| GET | `/api/agent/registry` | Returns registered agent list |
| POST | `/api/agent/register` | Register a new agent |
| POST | `/api/agent/callback` | Agent posts job result back to backend |

---

## 16. Analytics

Analytics summarize jobs and outputs. They do not depend on a rigid GL transaction model.

Aggregates include:
- Submission counts by status
- Output format distribution
- Recent job throughput
- Manager-scoped and employee-scoped summaries

---

## 17. Environment Variables

### Backend

```env
DATABASE_URL=postgresql+asyncpg://finflow:finflow@localhost:5433/finflow
REDIS_URL=redis://localhost:6379/0
AGENT_BASE_URL=http://localhost:8001
BACKEND_CALLBACK_URL=http://localhost:8000/api/agent/callback
AGENT_CALLBACK_SECRET=change-agent-callback-secret
UPLOAD_DIR=./storage/uploads
OUTPUT_DIR=./storage/outputs
JWT_SECRET_KEY=local-development-secret-change-before-deploy
FRONTEND_BASE_URL=http://localhost:5173
DEFAULT_ADMIN_EMAIL=admin@example.com
DEFAULT_ADMIN_PASSWORD=local-dev-admin-password
AGENT_EMAIL=
AGENT_PASSWORD=
AGENT_NAME=FinFlow Orchestrator
```

### Agent Service

```env
DATABASE_URL=postgresql://finflow:finflow@localhost:5433/finflow
BACKEND_CALLBACK_URL=http://localhost:8000/api/agent/callback
AGENT_CALLBACK_SECRET=change-agent-callback-secret
GROQ_API_KEY=your-groq-key
OUTPUT_DIR=./outputs
REDIS_URL=redis://localhost:6379/0
```

---

## 18. Deployment

Docker Compose starts PostgreSQL, Redis, the backend, the frontend, and the agent service together.

### Ports

| Service | Port |
|---|---|
| Frontend | `5173` |
| Backend API | `8000` |
| Agent service | `8001` |
| PostgreSQL | `5433` |
| Redis | `6379` |

### Setup

```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item agent-framework\new-agentic-project-data-cleaning-\.env.example agent-framework\new-agentic-project-data-cleaning-\.env
docker compose up --build
```

Agent service standalone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn finflow_agent.api:app --reload --port 8001
```

---

## 19. Design Constraints

These constraints are non-negotiable. Every future addition must be checked against them.

1. **Agent cap: 15 maximum.** Currently 7. Adding agents requires deliberate review, not convenience.
2. **No `AgentExecutor`.** LangGraph only. Any tutorial suggesting `AgentExecutor` must be mapped to LangGraph equivalents manually.
3. **Orchestrator plans upfront.** No agent makes routing decisions mid-DAG.
4. **`PipelineState` is in-memory.** Intermediate results are never written to the database.
5. **`partial` is `failed`.** Finance output is either correct or absent.
6. **Ingestion Agent is the only file-format-aware agent.** All other agents receive clean dataframes or text.
7. **Normalization lives in the Cleaning Agent.** No downstream agent does type coercion or format parsing.
8. **Filtering lives in the Filter Agent.** No other agent applies row or column selection.
9. **Display formatting lives in the Reporting Agent.** No other agent applies visual formatting.
10. **PNG is not a user-facing output format.** It is internal to the Visualization Agent only.
11. **`quarantined` is a visible submission status.** Nothing operational is silently buried.
12. **Keep the download flow pointed at the backend.** The user never downloads directly from the agent container.
13. **`submission_records` is the flexible structured data store for final output.** Intermediate step data goes to `audit_logs` only.
14. **`transaction_rows` is legacy compatibility only.** Do not extend it.
