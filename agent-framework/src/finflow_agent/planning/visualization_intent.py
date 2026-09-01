"""Visualization intent classification and DAG node construction.

Bridges the TriggerDetector with the plan compilation pipeline:
1. Classifies a user prompt for visualization intent using TriggerDetector.
2. Produces VisualizeIntent dataclass when trigger language is detected.
3. Builds VisualizationNode DAG nodes from intents, linking each to a calc step.

Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finflow_agent.execution.visualization_runner import (
    VisualizationNode,
    create_visualization_node,
)
from finflow_agent.planning.trigger_detector import TriggerDetector


# ---------------------------------------------------------------------------
# VisualizeIntent dataclass
# ---------------------------------------------------------------------------


@dataclass
class VisualizeIntentResult:
    """Represents a detected visualization intent from the user prompt.

    Attributes:
        chart_type_hint: Optional chart type suggestion from trigger language
            (e.g., "pie" from "pie chart"). None if no specific type detected.
        encoding_hints: Optional axis-to-field mapping hints. Empty dict by default.
        associated_operation_id: The operation_id of the calculation step this
            visualization should depend on.
    """

    chart_type_hint: str | None = None
    encoding_hints: dict[str, str] = field(default_factory=dict)
    associated_operation_id: str = ""


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

# Module-level singleton detector instance.
_detector = TriggerDetector()


def classify_visualization_intent(prompt: str) -> VisualizeIntentResult | None:
    """Classify a user prompt for visualization intent using TriggerDetector.

    Uses the TriggerDetector to detect trigger language in the prompt.
    Returns a VisualizeIntentResult if trigger language is found, None otherwise.

    Args:
        prompt: The user's input prompt text.

    Returns:
        VisualizeIntentResult if visualization trigger language detected,
        None otherwise (Req 13.1, 13.2: no visualization for non-trigger prompts).

    Requirements: 1.1, 1.4
    """
    if not prompt or not prompt.strip():
        return None

    result = _detector.detect(prompt)

    if not result.triggered:
        return None

    return VisualizeIntentResult(
        chart_type_hint=result.chart_type_hint,
        encoding_hints={},
        associated_operation_id="",
    )


# ---------------------------------------------------------------------------
# DAG Node Construction
# ---------------------------------------------------------------------------


def build_visualization_nodes_from_intent(
    intents: list[VisualizeIntentResult],
    calc_step_ids: list[str],
) -> list[VisualizationNode]:
    """Create visualization DAG nodes from intents, linking each to its associated calc step.

    Each VisualizeIntentResult is paired with a calculation step to form a
    VisualizationNode. If an intent has an explicit associated_operation_id,
    that is used as the depends_on reference. Otherwise, intents are paired
    positionally with calc_step_ids (first intent → first calc step, etc.).

    Args:
        intents: List of VisualizeIntentResult objects representing detected
            visualization intents.
        calc_step_ids: List of step_id strings for calculation steps that
            serve as data sources for visualizations.

    Returns:
        List of VisualizationNode objects in "pending" status, each linked
        to a source calculation step via depends_on.

    Requirements: 2.1, 5.2
    """
    if not intents:
        return []

    nodes: list[VisualizationNode] = []

    for idx, intent in enumerate(intents):
        # Determine which calc step this viz depends on.
        if intent.associated_operation_id:
            depends_on = intent.associated_operation_id
        elif idx < len(calc_step_ids):
            depends_on = calc_step_ids[idx]
        else:
            # Fall back to last available calc step if more intents than steps.
            depends_on = calc_step_ids[-1] if calc_step_ids else ""

        # Skip if no dependency can be established (no calc steps available).
        if not depends_on:
            continue

        node = create_visualization_node(
            depends_on=depends_on,
            operation_id=depends_on,
            chart_type=intent.chart_type_hint,
            encoding_hints=intent.encoding_hints,
        )
        nodes.append(node)

    return nodes
