import uuid

import pandas as pd


def test_ingestion_agent_rejects_file_path_param(tmp_path, bootstrap_agents):
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = IngestionAgent().execute(
        {"file_path": str(csv_path), "file_type": "csv"},
        {},
    )

    assert result.status == "failed"
    assert "resolved_file_path" in (result.error_message or "") or "file_path" in (result.error_message or "")


def test_engine_passes_visualization_artifacts_to_reporting():
    from finflow_agent.execution.engine import ExecutionEngine
    from finflow_agent.registry import registry, AgentSpec
    from finflow_agent.state import AgentResult, ExecutionPlan, PlanStep

    ingest_name = f"test_ingest_{uuid.uuid4().hex}"
    viz_name = f"test_viz_{uuid.uuid4().hex}"

    class FakeIngest:
        spec = AgentSpec(
            name=ingest_name,
            description="test ingest",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            return AgentResult(
                status="success",
                data=pd.DataFrame({"category": ["A", "B"], "amount": [10, 20]}),
            )

    class FakeViz:
        spec = AgentSpec(
            name=viz_name,
            description="test visualization",
            stage="visualize",
            accepts=["dataframe"],
            produces=["chart_spec"],
            params_schema={},
        )

        def execute(self, params, input_data):
            assert "input_dataframe" in input_data
            return AgentResult(
                status="success",
                data=None,
                artifacts={
                    "chart_test": {
                        "type": "bar",
                        "x": "category",
                        "y": "amount",
                        "title": "Amount by Category",
                    }
                },
            )

    class FakeReport:
        spec = AgentSpec(
            name="reporting_agent",
            description="fake report",
            stage="deliver",
            accepts=["dataframe"],
            produces=["file"],
            params_schema={},
        )

        received_input = None

        def execute(self, params, input_data):
            FakeReport.received_input = input_data
            assert "input_dataframe" in input_data
            assert "chart_artifacts" in input_data
            assert input_data["chart_artifacts"]
            return AgentResult(
                status="success",
                data=None,
                artifacts={"primary_output_path": "/tmp/fake_report.xlsx"},
            )

    registry.register(FakeIngest)
    registry.register(FakeViz)

    original_agent = registry._agents.get("reporting_agent")
    original_spec = registry._specs.get("reporting_agent")
    registry._agents["reporting_agent"] = FakeReport
    registry._specs["reporting_agent"] = FakeReport.spec

    try:
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id="ingest",
                    agent=ingest_name,
                    params={},
                    depends_on=[],
                    input_from=[],
                    output_key="df_ingested",
                ),
                PlanStep(
                    step_id="visualize",
                    agent=viz_name,
                    params={},
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="viz_artifacts",
                ),
                PlanStep(
                    step_id="report",
                    agent="reporting_agent",
                    params={},
                    depends_on=["ingest", "visualize"],
                    input_from=["df_ingested", "viz_artifacts"],
                    output_key="report_output",
                ),
            ]
        )

        result = ExecutionEngine().execute(plan)

        assert result["status"] == "complete"
        assert result["output_path"] == "/tmp/fake_report.xlsx"
        assert FakeReport.received_input is not None
        assert "chart_artifacts" in FakeReport.received_input
    finally:
        if original_agent is not None:
            registry._agents["reporting_agent"] = original_agent
        if original_spec is not None:
            registry._specs["reporting_agent"] = original_spec
