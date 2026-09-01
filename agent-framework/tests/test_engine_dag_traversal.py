"""DAG traversal smoke test for :class:`finflow_agent.engine.ExecutionEngine`.

Refactored from the original repository-root ``test_engine.py`` script into a
proper pytest. It verifies that the engine schedules and traverses a linear
three-stage DAG (ingestion -> cleaning -> reporting) to a ``complete`` result.

Deterministic stub agents are registered under uuid-suffixed names so the run
performs no real LLM calls or file I/O and never collides with sibling tests.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Type

import pandas as pd

from finflow_agent.engine import ExecutionEngine
from finflow_agent.registry import AgentSpec, registry
from finflow_agent.state import AgentResult, ExecutionPlan, PlanStep


@contextmanager
def _registered_agent(cls: Type):
    """Register a fake agent for the duration of a test and clean up after."""
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


def test_dag_traversal_completes_linear_pipeline():
    df_payload = pd.DataFrame({"A": [1], "B": [2]})
    call_order: list[str] = []

    ingest_name = _fake_name("ingestion_agent")
    clean_name = _fake_name("cleaning_agent")
    report_name = _fake_name("reporting_agent")

    def _make_agent(label: str, stage: str, name: str) -> Type:
        class _StubAgent:
            spec = AgentSpec(
                name=name,
                description=f"stub {label}",
                stage=stage,
                accepts=["dataframe"],
                produces=["dataframe"],
                params_schema={},
            )

            def execute(self, params, input_data):
                call_order.append(label)
                df = input_data.get("input_dataframe", df_payload)
                return AgentResult(status="success", data=df)

        _StubAgent.__name__ = f"Stub_{label}"
        return _StubAgent

    ingestion = _make_agent("ingestion", "ingest", ingest_name)
    cleaning = _make_agent("cleaning", "transform", clean_name)
    reporting = _make_agent("reporting", "deliver", report_name)

    plan = ExecutionPlan(
        steps=[
            PlanStep(step_id="step1", agent=ingest_name, depends_on=[], output_key="df_ingested"),
            PlanStep(
                step_id="step2",
                agent=clean_name,
                depends_on=["step1"],
                input_from=["df_ingested"],
                output_key="df_cleaned",
            ),
            PlanStep(
                step_id="step3",
                agent=report_name,
                depends_on=["step2"],
                input_from=["df_cleaned"],
                output_key="report_output",
            ),
        ]
    )

    with _registered_agent(ingestion), _registered_agent(cleaning), _registered_agent(reporting):
        result = ExecutionEngine().execute(plan)

    assert result["status"] == "complete", result
    # All three stages executed in dependency order.
    assert call_order == ["ingestion", "cleaning", "reporting"]
