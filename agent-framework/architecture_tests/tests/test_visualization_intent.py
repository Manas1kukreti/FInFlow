"""Unit tests for visualization_intent and plan_compiler_viz modules.

Tests cover:
- Intent classification with and without trigger language
- Node creation from intents
- Empty result when no visualization intent present
- Integration through compile_visualization_plan

Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
"""

import pytest

from finflow_agent.planning.visualization_intent import (
    VisualizeIntentResult,
    build_visualization_nodes_from_intent,
    classify_visualization_intent,
)
from finflow_agent.planning.plan_compiler_viz import compile_visualization_plan
from finflow_agent.execution.visualization_runner import VisualizationNode


# ---------------------------------------------------------------------------
# Tests for classify_visualization_intent
# ---------------------------------------------------------------------------


class TestClassifyVisualizationIntent:
    """Tests for classify_visualization_intent function."""

    def test_returns_intent_for_trigger_term(self):
        """Req 1.1: Trigger term 'chart' produces VisualizeIntentResult."""
        result = classify_visualization_intent("show me a chart of revenue")
        assert result is not None
        assert isinstance(result, VisualizeIntentResult)

    def test_returns_chart_type_hint_for_phrase(self):
        """Req 1.1: Phrase 'bar chart' produces chart_type_hint='bar'."""
        result = classify_visualization_intent("display a bar chart of sales")
        assert result is not None
        assert result.chart_type_hint == "bar"

    def test_returns_none_for_no_trigger(self):
        """Req 13.1, 13.2: No trigger language → None (no visualization)."""
        result = classify_visualization_intent("calculate the total revenue for Q1")
        assert result is None

    def test_returns_none_for_analytical_only(self):
        """Req 13.1: Analytical terms only → None."""
        result = classify_visualization_intent("show me the trend and distribution")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Empty prompt → None."""
        result = classify_visualization_intent("")
        assert result is None

    def test_returns_none_for_whitespace_only(self):
        """Whitespace-only prompt → None."""
        result = classify_visualization_intent("   \t\n  ")
        assert result is None

    def test_trigger_takes_precedence_over_analytical(self):
        """Req 1.4: Trigger + analytical terms → VisualizeIntentResult."""
        result = classify_visualization_intent("show a chart of the trend distribution")
        assert result is not None

    def test_pie_chart_hint(self):
        """Phrase 'pie chart' → chart_type_hint='pie'."""
        result = classify_visualization_intent("create a pie chart of expenses")
        assert result is not None
        assert result.chart_type_hint == "pie"

    def test_line_chart_hint(self):
        """Phrase 'line chart' → chart_type_hint='line'."""
        result = classify_visualization_intent("show a line chart of growth")
        assert result is not None
        assert result.chart_type_hint == "line"

    def test_scatter_plot_hint(self):
        """Phrase 'scatter plot' → chart_type_hint='scatter'."""
        result = classify_visualization_intent("make a scatter plot of x vs y")
        assert result is not None
        assert result.chart_type_hint == "scatter"

    def test_no_chart_type_hint_for_generic_term(self):
        """Generic 'graph' term → no specific chart_type_hint."""
        result = classify_visualization_intent("show me a graph of the data")
        assert result is not None
        assert result.chart_type_hint is None

    def test_substring_does_not_trigger(self):
        """Req 1.3: 'uncharted' contains 'chart' but should not trigger."""
        result = classify_visualization_intent("The uncharted territory was vast")
        assert result is None

    def test_encoding_hints_default_empty(self):
        """VisualizeIntentResult has empty encoding_hints by default."""
        result = classify_visualization_intent("visualize the data")
        assert result is not None
        assert result.encoding_hints == {}


# ---------------------------------------------------------------------------
# Tests for build_visualization_nodes_from_intent
# ---------------------------------------------------------------------------


class TestBuildVisualizationNodesFromIntent:
    """Tests for build_visualization_nodes_from_intent function."""

    def test_empty_intents_returns_empty_list(self):
        """No intents → no nodes."""
        nodes = build_visualization_nodes_from_intent([], ["calc_1"])
        assert nodes == []

    def test_single_intent_single_calc_step(self):
        """One intent + one calc step → one VisualizationNode."""
        intent = VisualizeIntentResult(
            chart_type_hint="bar",
            associated_operation_id="calc_1",
        )
        nodes = build_visualization_nodes_from_intent([intent], ["calc_1"])
        assert len(nodes) == 1
        assert nodes[0].depends_on == "calc_1"
        assert nodes[0].operation_id == "calc_1"
        assert nodes[0].chart_type == "bar"
        assert nodes[0].status == "pending"
        assert nodes[0].kind == "visualization"

    def test_multiple_intents_multiple_calc_steps(self):
        """Two intents + two calc steps → two nodes linked correctly."""
        intents = [
            VisualizeIntentResult(
                chart_type_hint="line",
                associated_operation_id="calc_1",
            ),
            VisualizeIntentResult(
                chart_type_hint="bar",
                associated_operation_id="calc_2",
            ),
        ]
        nodes = build_visualization_nodes_from_intent(intents, ["calc_1", "calc_2"])
        assert len(nodes) == 2
        assert nodes[0].depends_on == "calc_1"
        assert nodes[0].chart_type == "line"
        assert nodes[1].depends_on == "calc_2"
        assert nodes[1].chart_type == "bar"

    def test_intent_without_operation_id_uses_positional(self):
        """When associated_operation_id is empty, use positional calc step."""
        intent = VisualizeIntentResult(
            chart_type_hint="pie",
            associated_operation_id="",
        )
        nodes = build_visualization_nodes_from_intent([intent], ["step_abc"])
        assert len(nodes) == 1
        assert nodes[0].depends_on == "step_abc"

    def test_more_intents_than_calc_steps_uses_last(self):
        """When more intents than calc steps, extra intents use last calc step."""
        intents = [
            VisualizeIntentResult(associated_operation_id=""),
            VisualizeIntentResult(associated_operation_id=""),
            VisualizeIntentResult(associated_operation_id=""),
        ]
        nodes = build_visualization_nodes_from_intent(intents, ["calc_1", "calc_2"])
        assert len(nodes) == 3
        assert nodes[0].depends_on == "calc_1"
        assert nodes[1].depends_on == "calc_2"
        assert nodes[2].depends_on == "calc_2"

    def test_no_calc_steps_returns_empty(self):
        """Intents but no calc steps → empty (can't establish dependency)."""
        intent = VisualizeIntentResult(associated_operation_id="")
        nodes = build_visualization_nodes_from_intent([intent], [])
        assert nodes == []

    def test_encoding_hints_passed_through(self):
        """Encoding hints from intent are propagated to the node."""
        intent = VisualizeIntentResult(
            chart_type_hint="bar",
            encoding_hints={"x": "month", "y": "revenue"},
            associated_operation_id="calc_1",
        )
        nodes = build_visualization_nodes_from_intent([intent], ["calc_1"])
        assert len(nodes) == 1
        assert nodes[0].encoding_hints == {"x": "month", "y": "revenue"}

    def test_node_has_unique_step_id(self):
        """Each node gets a unique step_id."""
        intents = [
            VisualizeIntentResult(associated_operation_id="calc_1"),
            VisualizeIntentResult(associated_operation_id="calc_2"),
        ]
        nodes = build_visualization_nodes_from_intent(intents, ["calc_1", "calc_2"])
        assert nodes[0].step_id != nodes[1].step_id


# ---------------------------------------------------------------------------
# Tests for compile_visualization_plan
# ---------------------------------------------------------------------------


class TestCompileVisualizationPlan:
    """Tests for compile_visualization_plan high-level function."""

    def test_trigger_prompt_produces_nodes(self):
        """Prompt with trigger → VisualizationNode(s) returned."""
        nodes = compile_visualization_plan(
            prompt="show me a bar chart of revenue",
            calc_step_ids=["calc_1"],
        )
        assert len(nodes) == 1
        assert nodes[0].depends_on == "calc_1"
        assert nodes[0].chart_type == "bar"

    def test_no_trigger_returns_empty_list(self):
        """Req 13.1, 13.2: No trigger → empty list."""
        nodes = compile_visualization_plan(
            prompt="calculate the total revenue for Q1",
            calc_step_ids=["calc_1"],
        )
        assert nodes == []

    def test_empty_prompt_returns_empty_list(self):
        """Empty prompt → empty list."""
        nodes = compile_visualization_plan(prompt="", calc_step_ids=["calc_1"])
        assert nodes == []

    def test_no_calc_steps_returns_empty_list(self):
        """Trigger present but no calc steps → empty list."""
        nodes = compile_visualization_plan(
            prompt="show me a chart",
            calc_step_ids=[],
        )
        assert nodes == []

    def test_multiple_calc_steps_produces_multiple_nodes(self):
        """Trigger + multiple calc steps → one node per calc step."""
        nodes = compile_visualization_plan(
            prompt="visualize the results",
            calc_step_ids=["calc_1", "calc_2", "calc_3"],
        )
        assert len(nodes) == 3
        assert nodes[0].depends_on == "calc_1"
        assert nodes[1].depends_on == "calc_2"
        assert nodes[2].depends_on == "calc_3"

    def test_chart_type_hint_propagated(self):
        """Chart type hint from trigger phrase → node chart_type."""
        nodes = compile_visualization_plan(
            prompt="show me a pie chart of expenses",
            calc_step_ids=["calc_1"],
        )
        assert len(nodes) == 1
        assert nodes[0].chart_type == "pie"

    def test_generic_trigger_no_chart_type(self):
        """Generic 'graph' trigger → node chart_type is None."""
        nodes = compile_visualization_plan(
            prompt="show me a graph",
            calc_step_ids=["calc_step_0"],
        )
        assert len(nodes) == 1
        assert nodes[0].chart_type is None

    def test_analytical_only_terms_return_empty(self):
        """Analytical terms without trigger → empty list (Req 13.1)."""
        nodes = compile_visualization_plan(
            prompt="show me the trend and distribution breakdown",
            calc_step_ids=["calc_1"],
        )
        assert nodes == []

    def test_nodes_have_pending_status(self):
        """All returned nodes start with status 'pending'."""
        nodes = compile_visualization_plan(
            prompt="create a line chart",
            calc_step_ids=["calc_1"],
        )
        assert all(node.status == "pending" for node in nodes)

    def test_nodes_have_visualization_kind(self):
        """All returned nodes have kind='visualization'."""
        nodes = compile_visualization_plan(
            prompt="plot the data",
            calc_step_ids=["calc_1"],
        )
        assert all(node.kind == "visualization" for node in nodes)
