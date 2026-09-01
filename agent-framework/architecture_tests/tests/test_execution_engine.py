"""Deterministic unit tests for :mod:`finflow_agent.execution.engine`.

Covers the contract clauses task 9.1 added to ``ExecutionEngine.execute``:

* Per-step Pydantic params re-validation (req 4.1, 4.2, 10.6).
* Single-source ``input_dataframe`` resolution (req 4.3, 4.7, 5.5).
* Topological execution order (req 4.4).
* Stop on ``failed`` and ``partial`` agent results (req 4.5).
* Output-key storage discipline (req 4.6).
* Back-compat for agents without a registered ``params_model``
  (e.g. ``calculation_agent``).

All tests run on small in-memory DataFrames and deterministic stub agents
registered under ``uuid``-suffixed names so they never collide across tests.
The existing ``bootstrap_agents`` fixture seeds the real agent registry; the
tests below add additional fakes alongside it and undo the registration in a
``finally`` block so the registry's view stays clean for sibling tests.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import pytest

from finflow_agent.execution.engine import ExecutionEngine
from finflow_agent.registry import AgentSpec, registry
from finflow_agent.state import AgentResult, ExecutionPlan, PlanStep


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@contextmanager
def _registered_agent(cls: Type):
    """Register a fake agent class for the duration of a test.

    The registry stores the class under ``cls.spec.name`` and discards the
    entry on teardown so subsequent tests do not see leaked fakes. This is
    safe because every fake in this file uses a uuid-suffixed name.
    """
    registry.register(cls)
    try:
        yield cls
    finally:
        spec_name = cls.spec.name
        registry._agents.pop(spec_name, None)
        registry._specs.pop(spec_name, None)
        registry._param_models.pop(spec_name, None)


def _fake_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _build_envelope_step(
    step_id: str,
    agent: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    depends_on: Optional[List[str]] = None,
    input_from: Optional[List[str]] = None,
    output_key: Optional[str] = None,
) -> PlanStep:
    """Tiny PlanStep builder so tests stay focused on the assertions."""
    return PlanStep(
        step_id=step_id,
        agent=agent,
        params=params or {},
        depends_on=depends_on or [],
        input_from=input_from or [],
        output_key=output_key,
    )


# ---------------------------------------------------------------------------
# 1. Per-step param re-validation
# ---------------------------------------------------------------------------


def test_engine_re_validates_step_params_and_returns_failed_on_invalid_params(
    bootstrap_agents, tmp_path
):
    """Engine MUST re-validate ``step.params`` against the agent's
    registered Pydantic params model before invoking the agent.

    Builds a plan with a single ``ingestion_agent`` step whose
    ``params.file_type`` is ``"pdf"`` (rejected by the
    ``Literal["xlsx","xls","csv"]`` constraint on ``IngestionAgentParams``).
    The engine must short-circuit with ``status="failed"``, identify the
    offending step, and surface the Pydantic error message.
    """
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    plan = ExecutionPlan(
        steps=[
            _build_envelope_step(
                "ingest",
                "ingestion_agent",
                params={
                    "resolved_file_path": str(csv_path),
                    "file_type": "pdf",  # invalid literal -> ValidationError
                },
                output_key="df_ingested",
            ),
        ]
    )

    result = ExecutionEngine().execute(plan)

    assert result["status"] == "failed", result
    summary = result["summary"]
    assert summary["failed_step_id"] == "ingest"

    # The error message must reference the step and the param model so a
    # human reading the callback can locate the failure quickly.
    error = summary["error_message"]
    assert "ingest" in error
    assert "IngestionAgentParams" in error
    # Pydantic's literal-error message includes the field name.
    assert "file_type" in error


def test_engine_does_not_invoke_agent_when_params_validation_fails(
    bootstrap_agents,
):
    """When per-step params re-validation fails, ``agent.execute`` MUST
    NOT run. The fake ingest agent below records every call; the engine
    must leave that record empty."""
    fake_name = _fake_name("test_strict_params_ingest")

    class StrictParamsAgent:
        # Reuse the real ingestion params contract so file_type="pdf"
        # fails validation, but mark the class so we can detect any
        # accidental invocation.
        from finflow_agent.agents.ingestion_agent import IngestionAgentParams

        spec = AgentSpec(
            name=fake_name,
            description="Stub agent that should never execute under bad params.",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )
        params_model = IngestionAgentParams
        invocations: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            StrictParamsAgent.invocations.append(
                {"params": params, "input_data": input_data}
            )
            return AgentResult(status="success", data=pd.DataFrame())

    with _registered_agent(StrictParamsAgent):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step(
                    "ingest",
                    fake_name,
                    params={
                        "resolved_file_path": "x.csv",
                        "file_type": "pdf",  # rejected literal
                    },
                    output_key="df_ingested",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "failed"
    assert result["summary"]["failed_step_id"] == "ingest"
    # The decisive assertion: agent.execute was never called.
    assert StrictParamsAgent.invocations == []


# ---------------------------------------------------------------------------
# 2. Single-source ``input_dataframe`` resolution
# ---------------------------------------------------------------------------


def test_engine_input_dataframe_resolved_from_input_from_only(bootstrap_agents):
    """Two-step plan (ingest -> consumer). The consumer must receive
    ``input_data["input_dataframe"]`` exactly equal to the dataframe the
    upstream step produced, and NO other state-data keys must leak."""
    df_payload = pd.DataFrame(
        {"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]}
    )

    ingest_name = _fake_name("test_isolated_ingest")
    consumer_name = _fake_name("test_isolated_consumer")

    class IngestAgent:
        spec = AgentSpec(
            name=ingest_name,
            description="emit a known DataFrame",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(status="success", data=df_payload)

    class ConsumerAgent:
        spec = AgentSpec(
            name=consumer_name,
            description="capture the input_data passed to it",
            stage="transform",
            accepts=["dataframe"],
            produces=["dataframe"],
            params_schema={},
        )
        captured: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            ConsumerAgent.captured.append(input_data)
            return AgentResult(status="success", data=input_data["input_dataframe"])

    with _registered_agent(IngestAgent), _registered_agent(ConsumerAgent):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step(
                    "ingest",
                    ingest_name,
                    output_key="df_ingested",
                ),
                _build_envelope_step(
                    "consume",
                    consumer_name,
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_consumed",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "complete", result
    assert len(ConsumerAgent.captured) == 1

    received = ConsumerAgent.captured[0]
    # The dataframe MUST be the exact value the upstream agent emitted —
    # not a copy reconstructed from a serialized envelope.
    assert "input_dataframe" in received
    received_df = received["input_dataframe"]
    pd.testing.assert_frame_equal(received_df, df_payload)
    assert received_df is df_payload  # same object, never re-wrapped

    # No state-data key may leak into input_data. Only the canonical
    # input_dataframe lives there. chart_artifacts was not produced by
    # the upstream ingest step, so it must not appear.
    assert set(received.keys()) == {"input_dataframe"}, (
        f"input_data leaked extra keys: {set(received.keys()) - {'input_dataframe'}}"
    )


# ---------------------------------------------------------------------------
# 3. Stop on failed
# ---------------------------------------------------------------------------


def test_engine_stops_on_failed_step(bootstrap_agents):
    """When an agent returns ``status="failed"``, the engine must stop
    the DAG walk and return a failed callback that names the failed
    step. Any later step in topological order MUST NOT run."""
    df_payload = pd.DataFrame({"a": [1, 2]})

    ingest_name = _fake_name("test_failed_ingest")
    failer_name = _fake_name("test_failed_failer")
    sink_name = _fake_name("test_failed_sink")

    class IngestAgent:
        spec = AgentSpec(
            name=ingest_name,
            description="ok ingest",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(status="success", data=df_payload)

    class FailingAgent:
        spec = AgentSpec(
            name=failer_name,
            description="always fails",
            stage="transform",
            accepts=["dataframe"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(
                status="failed",
                error_message="deterministic test failure: simulated_clean_failure",
            )

    class SinkAgent:
        spec = AgentSpec(
            name=sink_name,
            description="must never run after a failed predecessor",
            stage="deliver",
            accepts=["dataframe"],
            produces=["file"],
            params_schema={},
        )
        invocations: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            SinkAgent.invocations.append(input_data)
            return AgentResult(
                status="success",
                artifacts={"primary_output_path": "/tmp/should_not_exist.xlsx"},
            )

    with (
        _registered_agent(IngestAgent),
        _registered_agent(FailingAgent),
        _registered_agent(SinkAgent),
    ):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step("ingest", ingest_name, output_key="df_ingested"),
                _build_envelope_step(
                    "fail",
                    failer_name,
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_failed",
                ),
                _build_envelope_step(
                    "sink",
                    sink_name,
                    depends_on=["fail"],
                    input_from=["df_failed"],
                    output_key="report_output",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "failed"
    summary = result["summary"]
    assert summary["failed_step_id"] == "fail"
    assert "simulated_clean_failure" in summary["error_message"]
    # Sink must never run after the upstream failure.
    assert SinkAgent.invocations == []


def test_engine_attributes_thrown_agent_exceptions_to_the_failing_step(
    bootstrap_agents,
):
    """If ``agent.execute`` raises, the engine must attribute the failure
    to that step instead of falling back to the first step."""
    df_payload = pd.DataFrame({"x": [1, 2, 3]})

    ingest_name = _fake_name("test_throwing_ingest")
    explode_name = _fake_name("test_throwing_explode")

    class IngestAgent:
        spec = AgentSpec(
            name=ingest_name,
            description="ok ingest",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(status="success", data=df_payload)

    class ExplodingAgent:
        spec = AgentSpec(
            name=explode_name,
            description="raises during execution",
            stage="transform",
            accepts=["dataframe"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            raise RuntimeError("boom during execute")

    with _registered_agent(IngestAgent), _registered_agent(ExplodingAgent):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step("ingest", ingest_name, output_key="df_ingested"),
                _build_envelope_step(
                    "explode",
                    explode_name,
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_exploded",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "failed"
    summary = result["summary"]
    assert summary["failed_step_id"] == "explode"
    assert "Unhandled exception in step 'explode'" in summary["error_message"]
    assert "boom during execute" in summary["error_message"]
    assert summary["step_statuses"]["explode"] == "failed"


# ---------------------------------------------------------------------------
# 4. Stop on partial
# ---------------------------------------------------------------------------


def test_engine_stops_on_partial_step(bootstrap_agents):
    """``partial`` is treated identically to ``failed`` for stopping
    purposes (req 4.5): the DAG walk halts and the callback is failed."""
    df_payload = pd.DataFrame({"a": [1, 2]})

    ingest_name = _fake_name("test_partial_ingest")
    partial_name = _fake_name("test_partial_partial")
    sink_name = _fake_name("test_partial_sink")

    class IngestAgent:
        spec = AgentSpec(
            name=ingest_name,
            description="ok ingest",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(status="success", data=df_payload)

    class PartialAgent:
        spec = AgentSpec(
            name=partial_name,
            description="returns partial",
            stage="transform",
            accepts=["dataframe"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(
                status="partial",
                error_message="some_rows_dropped_due_to_warning",
                data=df_payload,
            )

    class SinkAgent:
        spec = AgentSpec(
            name=sink_name,
            description="must not run after partial",
            stage="deliver",
            accepts=["dataframe"],
            produces=["file"],
            params_schema={},
        )
        invocations: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            SinkAgent.invocations.append(input_data)
            return AgentResult(
                status="success",
                artifacts={"primary_output_path": "/tmp/partial_must_not_exist.xlsx"},
            )

    with (
        _registered_agent(IngestAgent),
        _registered_agent(PartialAgent),
        _registered_agent(SinkAgent),
    ):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step("ingest", ingest_name, output_key="df_ingested"),
                _build_envelope_step(
                    "partial",
                    partial_name,
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_partial",
                ),
                _build_envelope_step(
                    "sink",
                    sink_name,
                    depends_on=["partial"],
                    input_from=["df_partial"],
                    output_key="report_output",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "failed"
    summary = result["summary"]
    assert summary["failed_step_id"] == "partial"
    assert summary["step_statuses"]["partial"] == "partial"
    assert SinkAgent.invocations == []


# ---------------------------------------------------------------------------
# 5. Output-key storage discipline
# ---------------------------------------------------------------------------


def test_engine_stores_envelope_under_output_key_only(bootstrap_agents):
    """After a successful run, only ``step.output_key`` (or ``step_id``
    when output_key is None) appears in ``state.data``. Specifically, the
    raw envelope must NOT leak into any other key.

    Verified indirectly via the consumer agent's input_data: only the
    explicitly-listed input_from keys may yield ``input_dataframe``, and
    no other state-data key may appear in the consumer's input_data.
    """
    df_payload = pd.DataFrame({"x": [10, 20]})

    ingest_name = _fake_name("test_output_key_ingest")
    transform_name = _fake_name("test_output_key_transform")
    sink_name = _fake_name("test_output_key_sink")

    class IngestAgent:
        spec = AgentSpec(
            name=ingest_name,
            description="ingest",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(status="success", data=df_payload)

    class TransformAgent:
        spec = AgentSpec(
            name=transform_name,
            description="transform",
            stage="transform",
            accepts=["dataframe"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            df = input_data["input_dataframe"]
            return AgentResult(status="success", data=df.assign(doubled=df["x"] * 2))

    class SinkAgent:
        spec = AgentSpec(
            name=sink_name,
            description="capture input_data and write a stub artifact",
            stage="deliver",
            accepts=["dataframe"],
            produces=["file"],
            params_schema={},
        )
        captured: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            SinkAgent.captured.append(input_data)
            return AgentResult(
                status="success",
                artifacts={"primary_output_path": "/tmp/ok.xlsx"},
            )

    with (
        _registered_agent(IngestAgent),
        _registered_agent(TransformAgent),
        _registered_agent(SinkAgent),
    ):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step("ingest", ingest_name, output_key="df_ingested"),
                _build_envelope_step(
                    "transform",
                    transform_name,
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_transformed",
                ),
                _build_envelope_step(
                    "sink",
                    sink_name,
                    depends_on=["transform"],
                    input_from=["df_transformed"],
                    output_key="report_output",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "complete", result

    # Sink saw exactly ONE state-data key, surfaced as input_dataframe.
    # No raw "df_ingested", "df_transformed", or "ingest"/"transform"
    # envelope leaked into the input_data.
    assert len(SinkAgent.captured) == 1
    sink_input = SinkAgent.captured[0]
    assert set(sink_input.keys()) == {"input_dataframe"}, (
        f"Sink input_data leaked extra keys: {sink_input.keys()}"
    )
    # The dataframe surfaced is the transform's output (has the new column).
    assert "doubled" in sink_input["input_dataframe"].columns


# ---------------------------------------------------------------------------
# 6. Topological order respected (diamond DAG)
# ---------------------------------------------------------------------------


def test_engine_topological_order_respected(bootstrap_agents):
    """3-step diamond plan: A -> B, A -> C, B+C -> D. The engine must
    execute A before both B and C, and D after both B and C.

    Each fake agent records the call order in a shared list. After the
    run, A is first, D is last, and B and C land between them. The diamond
    shape lets B and C land in either order so we don't over-specify the
    topological tie-break.
    """
    df_payload = pd.DataFrame({"value": [1, 2, 3, 4]})
    call_order: List[str] = []

    a_name = _fake_name("test_topo_a")
    b_name = _fake_name("test_topo_b")
    c_name = _fake_name("test_topo_c")
    d_name = _fake_name("test_topo_d")

    def _make_recorder(label: str, stage: str, name: str):
        class _R:
            spec = AgentSpec(
                name=name,
                description=f"recorder {label}",
                stage=stage,
                accepts=["dataframe"],
                produces=["dataframe"],
                params_schema={},
            )

            def execute(self, params, input_data):
                call_order.append(label)
                # Use input_dataframe when present so the dag is well-typed.
                df = input_data.get("input_dataframe", df_payload)
                return AgentResult(status="success", data=df)

        _R.__name__ = f"Recorder_{label}"
        return _R

    A = _make_recorder("A", "ingest", a_name)
    B = _make_recorder("B", "transform", b_name)
    C = _make_recorder("C", "transform", c_name)
    D = _make_recorder("D", "deliver", d_name)

    with _registered_agent(A), _registered_agent(B), _registered_agent(C), _registered_agent(D):
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step("a", a_name, output_key="df_a"),
                _build_envelope_step(
                    "b",
                    b_name,
                    depends_on=["a"],
                    input_from=["df_a"],
                    output_key="df_b",
                ),
                _build_envelope_step(
                    "c",
                    c_name,
                    depends_on=["a"],
                    input_from=["df_a"],
                    output_key="df_c",
                ),
                _build_envelope_step(
                    "d",
                    d_name,
                    depends_on=["b", "c"],
                    # D consumes from b in this test (single-source dataframe).
                    input_from=["df_b"],
                    output_key="report_output",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "complete", result
    assert len(call_order) == 4
    assert call_order[0] == "A", call_order
    assert call_order[-1] == "D", call_order
    # B and C run between A and D in either order.
    assert set(call_order[1:3]) == {"B", "C"}, call_order


# ---------------------------------------------------------------------------
# 7. Agent without a registered params_model (back-compat)
# ---------------------------------------------------------------------------


def test_engine_handles_agent_without_registered_params_model(bootstrap_agents):
    """Agents that intentionally do NOT declare a ``params_model``
    (today: ``calculation_agent``) must still execute end-to-end. The
    engine must skip per-step Pydantic re-validation rather than raising.
    """
    fake_name = _fake_name("test_no_params_model_agent")

    class NoParamsModelAgent:
        spec = AgentSpec(
            name=fake_name,
            description="like calculation_agent — no params_model declared",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )
        # Deliberately NO ``params_model`` attribute.
        invocations: List[Dict[str, Any]] = []

        def execute(self, params, input_data):
            NoParamsModelAgent.invocations.append(
                {"params": params, "input_data": input_data}
            )
            return AgentResult(
                status="success",
                data=pd.DataFrame({"a": [1]}),
            )

    # Sanity: registering the agent must NOT add a params model entry.
    with _registered_agent(NoParamsModelAgent):
        assert not registry.has_params_model(fake_name)

        # Pass arbitrary, non-validated params; the engine must forward
        # them to the agent unchanged.
        plan = ExecutionPlan(
            steps=[
                _build_envelope_step(
                    "ingest",
                    fake_name,
                    params={
                        "anything_goes": True,
                        "free_form_value": ["a", "b", 1, None],
                    },
                    output_key="df_anything",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

    assert result["status"] == "complete", result
    assert len(NoParamsModelAgent.invocations) == 1
    forwarded_params = NoParamsModelAgent.invocations[0]["params"]
    assert forwarded_params == {
        "anything_goes": True,
        "free_form_value": ["a", "b", 1, None],
    }
