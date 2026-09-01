"""Visualization-aware plan compiler.

High-level function that integrates TriggerDetector-based intent classification
with visualization DAG node construction. This module serves as the bridge
between the intent classification pipeline and the execution plan.

When a user prompt contains visualization trigger language, this module:
1. Detects the trigger via classify_visualization_intent
2. Builds VisualizationNode DAG entries linked to calculation steps
3. Returns the nodes for inclusion in the execution plan

When no trigger language is present, returns an empty list — ensuring backward
compatibility for jobs without visualization intent (Req 13.1, 13.2).

Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
"""

from __future__ import annotations

from finflow_agent.execution.visualization_runner import VisualizationNode
from finflow_agent.planning.visualization_intent import (
    VisualizeIntentResult,
    build_visualization_nodes_from_intent,
    classify_visualization_intent,
)


def compile_visualization_plan(
    prompt: str,
    calc_step_ids: list[str],
) -> list[VisualizationNode]:
    """Classify visualization intent and build viz DAG nodes if triggered.

    This is the top-level function that integrates the TriggerDetector with
    the plan compilation pipeline. It:
    1. Classifies the prompt for visualization trigger language
    2. If triggered, builds one VisualizationNode per calc step
    3. Returns empty list if no visualization intent detected

    Args:
        prompt: The user's input prompt text.
        calc_step_ids: List of step_id strings for calculation steps that
            provide data for visualizations.

    Returns:
        List of VisualizationNode objects to include in the execution plan.
        Empty list if no visualization intent is detected (Req 13.1, 13.2).

    Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
    """
    # Classify the prompt for visualization intent.
    intent = classify_visualization_intent(prompt)

    # No visualization trigger detected — return empty list.
    # This ensures backward compatibility: jobs without visualization
    # intent produce an empty visualizations array (Req 13.1, 13.2).
    if intent is None:
        return []

    # If no calc steps are available, visualization cannot proceed.
    if not calc_step_ids:
        return []

    # Build visualization nodes from the detected intent.
    # One node per calc step when a single intent covers all calculations.
    intents = [
        VisualizeIntentResult(
            chart_type_hint=intent.chart_type_hint,
            encoding_hints=intent.encoding_hints,
            associated_operation_id=step_id,
        )
        for step_id in calc_step_ids
    ]

    return build_visualization_nodes_from_intent(intents, calc_step_ids)
