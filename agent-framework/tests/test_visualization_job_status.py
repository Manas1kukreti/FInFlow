"""Unit tests for visualization failure isolation and job status logic.

Tests cover:
- Calc step status remains "success" when viz node fails/times out (Req 6.1)
- Job status "completed_with_warnings" when all calc succeed but ≥1 viz fails (Req 6.2)
- Job status "failed" when any calc step fails regardless of viz outcomes (Req 6.4)
- Failed viz node VisualizationSpec status "failed" with error ≤500 chars (Req 6.3)
- Continue executing remaining DAG nodes after viz failure (Req 6.3)
- One VisualizationSpec per viz node regardless of other nodes' outcomes (Req 5.3)
- Specs ordered by topological position in execution plan (Req 5.4)
- 30-second execution timeout for visualization nodes (Req 6.3)

Requirements: 5.3, 5.4, 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch


import pytest

from finflow_agent.execution.visualization.spec import VisualizationSpec
from finflow_agent.execution.visualization_job_status import (
    VisualizationJobResult,
    VisualizationJobStatusHandler,
)
from finflow_agent.execution.visualization_runner import (
    VisualizationNode,
    _VISUALIZATION_TIMEOUT_SECONDS,
    create_visualization_node,
    execute_visualization_nodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_operation_result(
    fields: list[dict] | None = None, rows: list[dict] | None = None
) -> dict:
    """Build a valid OperationResult dict."""
    if fields is None:
        fields = [
            {"id": "cat", "label": "Category", "data_type": "string", "role": "category"},
            {"id": "val", "label": "Revenue", "data_type": "float", "role": "measure"},
        ]
    result: dict = {"fields": fields}
    if rows is not None:
        result["rows"] = rows
    else:
        result["rows"] = [{"cat": "A", "val": 10.0}, {"cat": "B", "val": 20.0}]
    return result


def _make_calc_step_result(status: str = "success") -> dict:
    """Build a calc step result envelope."""
    return {
        "status": status,
        "agent": "calculation_agent",
        "summary": "Calculation completed",
        "metrics": {},
        "operations_applied": [],
        "warnings": [],
        "artifacts": {},
    }


def _make_ready_spec(operation_id: str = "op1", source_result_id: str = "calc1") -> VisualizationSpec:
    """Build a VisualizationSpec with status 'ready'."""
    return VisualizationSpec(
        operation_id=operation_id,
        source_result_id=source_result_id,
        status="ready",
        chart_type="bar",
        title="Test Chart",
        encoding={"x": "cat", "y": "val"},
        data=[{"cat": "A", "val": 10.0}],
    )


def _make_failed_spec(
    operation_id: str = "op1",
    source_result_id: str = "calc1",
    error: str = "Something went wrong",
) -> VisualizationSpec:
    """Build a VisualizationSpec with status 'failed'."""
    return VisualizationSpec(
        operation_id=operation_id,
        source_result_id=source_result_id,
        status="failed",
        chart_type="bar",
        title="Visualization Failed",
        encoding={},
        data=[],
        error=error,
    )


# ---------------------------------------------------------------------------
# Tests: Timeout constant
# ---------------------------------------------------------------------------


class TestTimeoutConstant:
    """Req 6.3: 30-second execution timeout."""

    def test_timeout_is_30_seconds(self):
        assert _VISUALIZATION_TIMEOUT_SECONDS == 30

    def test_handler_reports_30_second_timeout(self):
        handler = VisualizationJobStatusHandler()
        assert handler.timeout_seconds == 30


# ---------------------------------------------------------------------------
# Tests: Calc step status isolation (Req 6.1)
# ---------------------------------------------------------------------------


class TestCalcStatusIsolation:
    """Req 6.1: Calc step status remains 'success' when viz node fails/times out."""

    def test_calc_status_unchanged_when_viz_fails(self):
        """Calc step with status 'success' stays 'success' after viz failure."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        failed_specs = [_make_failed_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=failed_specs,
        )

        # Calc status must NOT have changed (Req 6.1)
        assert result.calc_step_statuses["calc1"] == "success"

    def test_calc_status_unchanged_when_multiple_viz_fail(self):
        """Multiple viz failures don't affect any calc step status."""
        handler = VisualizationJobStatusHandler()
        calc_results = {
            "calc1": _make_calc_step_result("success"),
            "calc2": _make_calc_step_result("success"),
        }
        failed_specs = [
            _make_failed_spec(operation_id="op1", source_result_id="calc1"),
            _make_failed_spec(operation_id="op2", source_result_id="calc2"),
        ]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=failed_specs,
        )

        assert result.calc_step_statuses["calc1"] == "success"
        assert result.calc_step_statuses["calc2"] == "success"

    def test_verify_isolation_returns_true_when_statuses_unchanged(self):
        """verify_calc_status_isolation returns True when no statuses changed."""
        handler = VisualizationJobStatusHandler()
        before = {"calc1": "success", "calc2": "success"}
        after = {
            "calc1": _make_calc_step_result("success"),
            "calc2": _make_calc_step_result("success"),
        }

        assert handler.verify_calc_status_isolation(before, after) is True

    def test_verify_isolation_returns_false_when_status_changed(self):
        """verify_calc_status_isolation detects a status mutation."""
        handler = VisualizationJobStatusHandler()
        before = {"calc1": "success"}
        after = {"calc1": _make_calc_step_result("failed")}

        assert handler.verify_calc_status_isolation(before, after) is False

    def test_verify_isolation_returns_false_when_step_removed(self):
        """verify_calc_status_isolation detects a removed step."""
        handler = VisualizationJobStatusHandler()
        before = {"calc1": "success"}
        after = {}

        assert handler.verify_calc_status_isolation(before, after) is False

    def test_calc_status_preserved_with_timeout_viz(self):
        """Calc step status unchanged when viz times out."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        timeout_spec = handler.build_timeout_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
        )

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=[timeout_spec],
        )

        assert result.calc_step_statuses["calc1"] == "success"

    def test_execution_runner_preserves_calc_status_on_viz_failure(self):
        """End-to-end: execute_visualization_nodes with a failing viz
        does not mutate the original step_results."""
        step_results = {"calc1": _make_calc_step_result("success")}
        # Snapshot statuses before
        statuses_before = {k: v["status"] for k, v in step_results.items()}

        node = create_visualization_node(
            depends_on="calc_nonexistent",  # Will fail dependency resolution
            operation_id="op1",
            chart_type="bar",
        )

        execute_visualization_nodes(
            [node],
            step_results,
            {},
        )

        # Verify step_results unchanged (Req 6.1)
        statuses_after = {k: v["status"] for k, v in step_results.items()}
        assert statuses_before == statuses_after


# ---------------------------------------------------------------------------
# Tests: Job status "completed_with_warnings" (Req 6.2)
# ---------------------------------------------------------------------------


class TestJobStatusCompletedWithWarnings:
    """Req 6.2: Job status = 'completed_with_warnings' when all calc succeed but ≥1 viz fails."""

    def test_all_calc_success_one_viz_fail(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [_make_failed_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "completed_with_warnings"

    def test_all_calc_success_multiple_viz_fail(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {
            "calc1": _make_calc_step_result("success"),
            "calc2": _make_calc_step_result("success"),
        }
        specs = [
            _make_failed_spec(operation_id="op1"),
            _make_failed_spec(operation_id="op2"),
        ]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "completed_with_warnings"

    def test_all_calc_success_mixed_viz_results(self):
        """Some viz succeed, some fail — still completed_with_warnings."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [
            _make_ready_spec(operation_id="op_ok"),
            _make_failed_spec(operation_id="op_bad"),
        ]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "completed_with_warnings"

    def test_completed_with_warnings_includes_warning_message(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [_make_failed_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert len(result.warnings) > 0
        assert "1 visualization(s) failed" in result.warnings[0]


# ---------------------------------------------------------------------------
# Tests: Job status "failed" (Req 6.4)
# ---------------------------------------------------------------------------


class TestJobStatusFailed:
    """Req 6.4: Job status = 'failed' when any calc step fails regardless of viz outcomes."""

    def test_one_calc_fails_no_viz(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("failed")}
        specs = []

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "failed"

    def test_one_calc_fails_viz_succeeds(self):
        """Calc failure overrides viz success."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("failed")}
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "failed"

    def test_one_calc_fails_viz_also_fails(self):
        """Both calc and viz fail → job status is 'failed' (calc takes priority)."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("failed")}
        specs = [_make_failed_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "failed"

    def test_partial_calc_treated_as_failure(self):
        """Partial status on calc is treated as failure."""
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("partial")}
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "failed"

    def test_mixed_calc_statuses_one_fails(self):
        """Multiple calc steps, one fails → job fails."""
        handler = VisualizationJobStatusHandler()
        calc_results = {
            "calc1": _make_calc_step_result("success"),
            "calc2": _make_calc_step_result("failed"),
        }
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "failed"


# ---------------------------------------------------------------------------
# Tests: Job status "complete" (all succeed)
# ---------------------------------------------------------------------------


class TestJobStatusComplete:
    """All calc succeed and all viz succeed → job status = 'complete'."""

    def test_all_succeed(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "complete"

    def test_all_calc_succeed_no_viz(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = []

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.job_status == "complete"

    def test_no_warnings_when_all_succeed(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert result.warnings == []


# ---------------------------------------------------------------------------
# Tests: Failed viz spec error capped at 500 chars (Req 6.3)
# ---------------------------------------------------------------------------


class TestFailedVizSpecError:
    """Req 6.3: Failed viz node VisualizationSpec error ≤500 chars."""

    def test_error_under_500_chars_unchanged(self):
        handler = VisualizationJobStatusHandler()
        error = "Short error message"
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
            error=error,
        )

        assert spec.error == error
        assert spec.status == "failed"

    def test_error_exactly_500_chars_unchanged(self):
        handler = VisualizationJobStatusHandler()
        error = "x" * 500
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
            error=error,
        )

        assert len(spec.error) == 500

    def test_error_over_500_chars_truncated(self):
        handler = VisualizationJobStatusHandler()
        error = "y" * 1000
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
            error=error,
        )

        assert len(spec.error) == 500

    def test_empty_error_gets_default_message(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
            error="",
        )

        assert spec.error == "Unknown visualization failure."

    def test_failed_spec_has_empty_data_and_encoding(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
            error="Test error",
        )

        assert spec.data == []
        assert spec.encoding == {}

    def test_failed_spec_chart_type_defaults_to_auto(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_failed_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type=None,
            error="Test error",
        )

        assert spec.chart_type == "auto"


# ---------------------------------------------------------------------------
# Tests: Timeout spec (Req 6.3)
# ---------------------------------------------------------------------------


class TestTimeoutSpec:
    """Req 6.3: Timeout produces a 'failed' spec with timeout message."""

    def test_timeout_spec_has_failed_status(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_timeout_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
        )

        assert spec.status == "failed"

    def test_timeout_spec_error_mentions_30_seconds(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_timeout_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
        )

        assert "30 seconds" in spec.error

    def test_timeout_spec_error_under_500_chars(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_timeout_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
        )

        assert len(spec.error) <= 500

    def test_timeout_spec_has_empty_data(self):
        handler = VisualizationJobStatusHandler()
        spec = handler.build_timeout_viz_spec(
            operation_id="op1",
            source_result_id="calc1",
            chart_type="bar",
        )

        assert spec.data == []
        assert spec.encoding == {}


# ---------------------------------------------------------------------------
# Tests: Continue executing remaining nodes after viz failure (Req 6.3)
# ---------------------------------------------------------------------------


class TestContinueAfterVizFailure:
    """Req 6.3: Continue executing remaining DAG nodes after viz failure."""

    def test_remaining_nodes_execute_after_one_fails(self):
        """When one viz node fails (bad dependency), others still execute."""
        nodes = [
            create_visualization_node(
                depends_on="calc_missing",  # Will fail — no such step
                operation_id="op_fail",
                chart_type="bar",
            ),
            create_visualization_node(
                depends_on="calc_ok",
                operation_id="op_ok",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
            ),
        ]

        step_results = {
            "calc_ok": _make_calc_step_result("success"),
        }
        pipeline_data = {
            "calc_ok": _make_operation_result(),
        }

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        # Both nodes must produce a spec (Req 5.3)
        assert len(specs) == 2
        # First node failed (missing dependency)
        assert specs[0].status == "failed"
        # Second node succeeded
        assert specs[1].status == "ready"

    def test_all_nodes_execute_even_if_first_throws(self):
        """Unhandled exception in first node doesn't prevent others from running."""
        nodes = [
            create_visualization_node(
                depends_on="calc1",
                operation_id="op_throw",
                chart_type="bar",
            ),
            create_visualization_node(
                depends_on="calc1",
                operation_id="op_ok",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
            ),
        ]

        step_results = {
            "calc1": _make_calc_step_result("success"),
        }
        # First node gets invalid data that will cause executor to fail
        # Second node gets valid data
        pipeline_data = {
            "calc1": _make_operation_result(),
        }

        # Patch the executor to throw for the first node but work for the second
        original_execute = None
        call_count = [0]

        from finflow_agent.execution.visualization.executor import VisualizationExecutor

        original_execute = VisualizationExecutor.execute

        def patched_execute(self, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("operation_id") == "op_throw":
                raise RuntimeError("Simulated executor crash")
            return original_execute(self, *args, **kwargs)

        with patch.object(VisualizationExecutor, "execute", patched_execute):
            specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        # Both nodes produced specs
        assert len(specs) == 2
        assert specs[0].status == "failed"
        assert "unexpected error" in specs[0].error.lower() or "Simulated executor crash" in specs[0].error
        assert specs[1].status == "ready"


# ---------------------------------------------------------------------------
# Tests: One VisualizationSpec per viz node (Req 5.3)
# ---------------------------------------------------------------------------


class TestOneSpecPerNode:
    """Req 5.3: Produce one VisualizationSpec per viz node regardless of other nodes' outcomes."""

    def test_n_nodes_produce_n_specs(self):
        """N nodes → exactly N specs."""
        n = 5
        nodes = [
            create_visualization_node(
                depends_on=f"calc{i}",
                operation_id=f"op{i}",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
            )
            for i in range(n)
        ]

        step_results = {
            f"calc{i}": _make_calc_step_result("success")
            for i in range(n)
        }
        pipeline_data = {
            f"calc{i}": _make_operation_result()
            for i in range(n)
        }

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert len(specs) == n

    def test_all_failing_nodes_still_produce_specs(self):
        """Even when all nodes fail, each one produces exactly one spec."""
        nodes = [
            create_visualization_node(
                depends_on=f"missing_calc{i}",
                operation_id=f"op{i}",
                chart_type="bar",
            )
            for i in range(3)
        ]

        specs = execute_visualization_nodes(nodes, {}, {})

        assert len(specs) == 3
        for spec in specs:
            assert spec.status == "failed"

    def test_mixed_outcomes_still_produce_one_spec_per_node(self):
        """Mixed successes and failures: each node still produces exactly one spec."""
        nodes = [
            create_visualization_node(depends_on="calc_ok", operation_id="op1",
                                      chart_type="bar", encoding_hints={"x": "cat", "y": "val"}),
            create_visualization_node(depends_on="calc_missing", operation_id="op2",
                                      chart_type="bar"),
            create_visualization_node(depends_on="calc_ok", operation_id="op3",
                                      chart_type="bar", encoding_hints={"x": "cat", "y": "val"}),
        ]

        step_results = {"calc_ok": _make_calc_step_result("success")}
        pipeline_data = {"calc_ok": _make_operation_result()}

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert len(specs) == 3
        assert specs[0].status == "ready"
        assert specs[1].status == "failed"
        assert specs[2].status == "ready"


# ---------------------------------------------------------------------------
# Tests: Specs ordered by topological position (Req 5.4)
# ---------------------------------------------------------------------------


class TestSpecsTopologicalOrder:
    """Req 5.4: Return specs ordered by topological position in execution plan."""

    def test_order_matches_input_order(self):
        """Specs are returned in same order as input nodes list."""
        nodes = [
            create_visualization_node(
                depends_on="calc1",
                operation_id=f"op_{i}",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
                step_id=f"viz_ordered_{i}",
            )
            for i in range(4)
        ]

        step_results = {"calc1": _make_calc_step_result("success")}
        pipeline_data = {"calc1": _make_operation_result()}

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert len(specs) == 4
        for i, spec in enumerate(specs):
            assert spec.operation_id == f"op_{i}"

    def test_order_preserved_even_with_failures(self):
        """Order preserved regardless of which nodes fail."""
        nodes = [
            create_visualization_node(depends_on="calc1", operation_id="first",
                                      chart_type="bar", encoding_hints={"x": "cat", "y": "val"}),
            create_visualization_node(depends_on="missing", operation_id="second",
                                      chart_type="bar"),
            create_visualization_node(depends_on="calc1", operation_id="third",
                                      chart_type="bar", encoding_hints={"x": "cat", "y": "val"}),
        ]

        step_results = {"calc1": _make_calc_step_result("success")}
        pipeline_data = {"calc1": _make_operation_result()}

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert specs[0].operation_id == "first"
        assert specs[1].operation_id == "second"
        assert specs[2].operation_id == "third"


# ---------------------------------------------------------------------------
# Tests: VisualizationJobResult container
# ---------------------------------------------------------------------------


class TestVisualizationJobResult:
    """Test the VisualizationJobResult dataclass structure."""

    def test_result_contains_all_fields(self):
        handler = VisualizationJobStatusHandler()
        calc_results = {"calc1": _make_calc_step_result("success")}
        specs = [_make_ready_spec()]

        result = handler.determine_job_status(
            calc_step_results=calc_results,
            visualization_specs=specs,
        )

        assert isinstance(result, VisualizationJobResult)
        assert result.job_status == "complete"
        assert len(result.visualization_specs) == 1
        assert result.calc_step_statuses == {"calc1": "success"}
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Tests: Integration — execute_visualization_nodes with error ≤500 chars
# ---------------------------------------------------------------------------


class TestRunnerErrorCapping:
    """Verify that the runner produces errors ≤500 chars in VisualizationSpec."""

    def test_failed_dependency_error_under_500_chars(self):
        """When dependency resolution fails, the error in spec is ≤500 chars."""
        node = create_visualization_node(
            depends_on="a_very_long_step_name_" + "x" * 500,
            operation_id="op1",
            chart_type="bar",
        )

        specs = execute_visualization_nodes([node], {}, {})

        assert len(specs) == 1
        assert specs[0].status == "failed"
        assert len(specs[0].error) <= 500
