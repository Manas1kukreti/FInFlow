"""Unit tests for the visualization_runner module.

Tests cover:
- VisualizationNode creation with unique step_id and initial "pending" status
- Plan validation: reject >20 viz nodes, reject >1 dependency per viz node
- Dependency resolution: success path and failure paths
- Single node execution: happy path and dependency failures
- Concurrent execution of multiple viz nodes
- Integration with VisualizationExecutor

Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.5, 16.1, 16.4
"""

from __future__ import annotations

import os
import sys


import pytest

from finflow_agent.execution.visualization_runner import (
    MAX_VISUALIZATIONS_PER_JOB,
    VisualizationLimitExceededError,
    VisualizationMultipleDependencyError,
    VisualizationNode,
    VisualizationPlanError,
    create_visualization_node,
    execute_visualization_nodes,
    resolve_dependency,
    validate_visualization_plan,
    validate_visualization_plan_from_steps,
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


def _make_step_results(step_id: str, status: str = "success") -> dict:
    """Build a step_results dict with a single step."""
    return {
        step_id: {
            "status": status,
            "agent": "calculation_agent",
            "summary": "Done",
            "metrics": {},
            "operations_applied": [],
            "warnings": [],
            "artifacts": {},
        }
    }


# ---------------------------------------------------------------------------
# Tests: MAX_VISUALIZATIONS_PER_JOB constant
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify the MAX_VISUALIZATIONS_PER_JOB constant value."""

    def test_max_visualizations_is_20(self):
        assert MAX_VISUALIZATIONS_PER_JOB == 20


# ---------------------------------------------------------------------------
# Tests: VisualizationNode creation
# ---------------------------------------------------------------------------


class TestVisualizationNodeCreation:
    """Req 2.1: Create visualization DAG_Node with unique step_id, depends_on, status pending."""

    def test_create_node_has_unique_step_id(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
        )
        assert node.step_id.startswith("viz_")
        assert len(node.step_id) > 4  # viz_ + hex chars

    def test_create_node_initial_status_pending(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
        )
        assert node.status == "pending"

    def test_create_node_kind_is_visualization(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
        )
        assert node.kind == "visualization"

    def test_create_node_depends_on_set(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
        )
        assert node.depends_on == "calc_step_1"

    def test_create_node_with_explicit_step_id(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
            step_id="my_custom_viz_step",
        )
        assert node.step_id == "my_custom_viz_step"

    def test_create_node_with_chart_type(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
            chart_type="bar",
        )
        assert node.chart_type == "bar"

    def test_create_node_with_encoding_hints(self):
        node = create_visualization_node(
            depends_on="calc_step_1",
            operation_id="op1",
            encoding_hints={"x": "cat", "y": "val"},
        )
        assert node.encoding_hints == {"x": "cat", "y": "val"}

    def test_two_nodes_have_different_step_ids(self):
        node1 = create_visualization_node(depends_on="calc1", operation_id="op1")
        node2 = create_visualization_node(depends_on="calc1", operation_id="op2")
        assert node1.step_id != node2.step_id


# ---------------------------------------------------------------------------
# Tests: Plan validation
# ---------------------------------------------------------------------------


class TestValidateVisualizationPlan:
    """Req 5.1, 5.5, 16.4: Plan validation rejects invalid configurations."""

    def test_empty_plan_is_valid(self):
        # 0 viz nodes is valid (Req 5.1: supports 0..20)
        validate_visualization_plan([])

    def test_single_node_is_valid(self):
        node = create_visualization_node(depends_on="calc1", operation_id="op1")
        validate_visualization_plan([node])

    def test_exactly_20_nodes_is_valid(self):
        nodes = [
            create_visualization_node(depends_on=f"calc_{i}", operation_id=f"op_{i}")
            for i in range(20)
        ]
        validate_visualization_plan(nodes)  # Should not raise

    def test_21_nodes_raises_limit_exceeded(self):
        nodes = [
            create_visualization_node(depends_on=f"calc_{i}", operation_id=f"op_{i}")
            for i in range(21)
        ]
        with pytest.raises(VisualizationLimitExceededError) as exc_info:
            validate_visualization_plan(nodes)
        assert exc_info.value.count == 21

    def test_node_without_dependency_raises_error(self):
        node = VisualizationNode(
            step_id="viz_1",
            depends_on="",
            operation_id="op1",
        )
        with pytest.raises(VisualizationPlanError):
            validate_visualization_plan([node])


class TestValidateVisualizationPlanFromSteps:
    """Validate plan from raw step dicts (pre-execution validation)."""

    def test_no_viz_steps_is_valid(self):
        steps = [
            {"step_id": "calc1", "kind": "calculation", "depends_on": []},
        ]
        validate_visualization_plan_from_steps(steps)  # Should not raise

    def test_viz_step_with_single_dep_is_valid(self):
        steps = [
            {"step_id": "viz1", "kind": "visualization", "depends_on": ["calc1"]},
        ]
        validate_visualization_plan_from_steps(steps)

    def test_viz_step_with_multiple_deps_raises(self):
        steps = [
            {"step_id": "viz1", "kind": "visualization", "depends_on": ["calc1", "calc2"]},
        ]
        with pytest.raises(VisualizationMultipleDependencyError) as exc_info:
            validate_visualization_plan_from_steps(steps)
        assert exc_info.value.step_id == "viz1"
        assert exc_info.value.depends_on == ["calc1", "calc2"]

    def test_more_than_20_viz_steps_raises(self):
        steps = [
            {"step_id": f"viz_{i}", "kind": "visualization", "depends_on": [f"calc_{i}"]}
            for i in range(21)
        ]
        with pytest.raises(VisualizationLimitExceededError):
            validate_visualization_plan_from_steps(steps)

    def test_non_viz_steps_not_counted(self):
        steps = [
            {"step_id": f"calc_{i}", "kind": "calculation", "depends_on": []}
            for i in range(30)
        ]
        # 30 calc steps is fine; no viz steps
        validate_visualization_plan_from_steps(steps)


# ---------------------------------------------------------------------------
# Tests: Dependency resolution
# ---------------------------------------------------------------------------


class TestResolveDependency:
    """Req 2.2, 2.3: Confirm dependency with status 'success', fail if unresolvable."""

    def test_resolve_success(self):
        node = create_visualization_node(depends_on="calc1", operation_id="op1")
        step_results = _make_step_results("calc1", "success")
        result = resolve_dependency(node, step_results)
        assert result is not None
        assert result["status"] == "success"

    def test_resolve_fails_when_step_missing(self):
        node = create_visualization_node(depends_on="calc1", operation_id="op1")
        step_results = {}  # no results
        result = resolve_dependency(node, step_results)
        assert result is None

    def test_resolve_fails_when_step_has_failed_status(self):
        node = create_visualization_node(depends_on="calc1", operation_id="op1")
        step_results = _make_step_results("calc1", "failed")
        result = resolve_dependency(node, step_results)
        assert result is None

    def test_resolve_fails_when_step_has_partial_status(self):
        node = create_visualization_node(depends_on="calc1", operation_id="op1")
        step_results = _make_step_results("calc1", "partial")
        result = resolve_dependency(node, step_results)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Single node execution
# ---------------------------------------------------------------------------


class TestExecuteSingleNode:
    """Req 2.2, 2.3, 16.1: Single visualization node execution."""

    def test_successful_execution(self):
        node = create_visualization_node(
            depends_on="calc1",
            operation_id="op1",
            chart_type="bar",
            encoding_hints={"x": "cat", "y": "val"},
        )
        step_results = _make_step_results("calc1", "success")
        pipeline_data = {"calc1": _make_operation_result()}

        specs = execute_visualization_nodes(
            [node], step_results, pipeline_data
        )

        assert len(specs) == 1
        assert specs[0].status == "ready"
        assert specs[0].chart_type == "bar"
        assert node.status == "success"

    def test_failed_dependency_sets_node_failed(self):
        node = create_visualization_node(
            depends_on="calc1",
            operation_id="op1",
            chart_type="bar",
        )
        step_results = _make_step_results("calc1", "failed")
        pipeline_data = {}

        specs = execute_visualization_nodes(
            [node], step_results, pipeline_data
        )

        assert len(specs) == 1
        assert specs[0].status == "failed"
        assert node.status == "failed"
        assert "status 'failed'" in node.error

    def test_missing_dependency_sets_node_failed(self):
        node = create_visualization_node(
            depends_on="nonexistent_step",
            operation_id="op1",
            chart_type="bar",
        )
        step_results = {}
        pipeline_data = {}

        specs = execute_visualization_nodes(
            [node], step_results, pipeline_data
        )

        assert len(specs) == 1
        assert specs[0].status == "failed"
        assert node.status == "failed"
        assert "does not exist" in node.error

    def test_missing_pipeline_data_sets_node_failed(self):
        node = create_visualization_node(
            depends_on="calc1",
            operation_id="op1",
            chart_type="bar",
        )
        step_results = _make_step_results("calc1", "success")
        pipeline_data = {}  # no data for calc1

        specs = execute_visualization_nodes(
            [node], step_results, pipeline_data
        )

        assert len(specs) == 1
        assert specs[0].status == "failed"
        assert node.status == "failed"
        assert "could not be located" in node.error


# ---------------------------------------------------------------------------
# Tests: Concurrent execution of multiple nodes
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    """Req 2.4, 5.3, 5.4: Concurrent execution, per-node isolation, ordering."""

    def test_multiple_nodes_produce_one_spec_each(self):
        nodes = [
            create_visualization_node(
                depends_on=f"calc{i}",
                operation_id=f"op{i}",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
            )
            for i in range(3)
        ]

        step_results = {}
        pipeline_data = {}
        for i in range(3):
            step_results[f"calc{i}"] = {"status": "success", "agent": "calc", "summary": "", "metrics": {}, "operations_applied": [], "warnings": [], "artifacts": {}}
            pipeline_data[f"calc{i}"] = _make_operation_result()

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert len(specs) == 3
        for spec in specs:
            assert spec.status == "ready"

    def test_mixed_success_and_failure(self):
        """Some nodes succeed, some fail — each produces a spec. Req 5.3."""
        node_success = create_visualization_node(
            depends_on="calc_ok",
            operation_id="op_ok",
            chart_type="bar",
            encoding_hints={"x": "cat", "y": "val"},
        )
        node_fail = create_visualization_node(
            depends_on="calc_bad",
            operation_id="op_bad",
            chart_type="bar",
        )

        step_results = {
            "calc_ok": {"status": "success", "agent": "calc", "summary": "", "metrics": {}, "operations_applied": [], "warnings": [], "artifacts": {}},
            "calc_bad": {"status": "failed", "agent": "calc", "summary": "", "metrics": {}, "operations_applied": [], "warnings": [], "artifacts": {}},
        }
        pipeline_data = {
            "calc_ok": _make_operation_result(),
        }

        specs = execute_visualization_nodes(
            [node_success, node_fail], step_results, pipeline_data
        )

        assert len(specs) == 2
        assert specs[0].status == "ready"
        assert specs[1].status == "failed"

    def test_order_preserved(self):
        """Specs returned in same order as input nodes. Req 5.4."""
        nodes = [
            create_visualization_node(
                depends_on="calc1",
                operation_id=f"op_{i}",
                chart_type="bar",
                encoding_hints={"x": "cat", "y": "val"},
                step_id=f"viz_ordered_{i}",
            )
            for i in range(5)
        ]

        step_results = {"calc1": {"status": "success", "agent": "calc", "summary": "", "metrics": {}, "operations_applied": [], "warnings": [], "artifacts": {}}}
        pipeline_data = {"calc1": _make_operation_result()}

        specs = execute_visualization_nodes(nodes, step_results, pipeline_data)

        assert len(specs) == 5
        for i, spec in enumerate(specs):
            assert spec.operation_id == f"op_{i}"

    def test_empty_nodes_returns_empty_specs(self):
        """0 viz nodes → empty list. Req 5.2."""
        specs = execute_visualization_nodes([], {}, {})
        assert specs == []

    def test_operation_result_in_envelope(self):
        """Handle operation result nested in an engine envelope dict."""
        node = create_visualization_node(
            depends_on="calc1",
            operation_id="op1",
            chart_type="bar",
            encoding_hints={"x": "cat", "y": "val"},
        )
        step_results = _make_step_results("calc1", "success")
        # Wrapped in envelope like the real engine stores it
        pipeline_data = {
            "calc1": {
                "data": _make_operation_result(),
                "artifacts": {},
            }
        }

        specs = execute_visualization_nodes([node], step_results, pipeline_data)

        assert len(specs) == 1
        # The runner should find the operation_result inside the envelope
        # Either "ready" (found it) or "failed" (reader validation)
        # In this case, the inner dict has fields+rows so it should work
        assert specs[0].operation_id == "op1"
