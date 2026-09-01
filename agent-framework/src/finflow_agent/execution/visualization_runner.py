"""Visualization DAG runner — handles visualization node lifecycle in the execution plan.

This module implements visualization-specific DAG logic that integrates with the
main ExecutionEngine. It manages the creation, validation, dependency resolution,
and concurrent execution of visualization DAG nodes.

Key responsibilities:
- Plan validation: reject plans with >20 viz nodes or multi-dependency viz nodes
- Dependency resolution: confirm source calc step completed with status "success"
- Concurrent execution: run independent viz nodes in parallel via asyncio
- Integration with VisualizationExecutor to produce VisualizationSpecs

Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.5, 16.1, 16.4
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from finflow_agent.execution.visualization.executor import VisualizationExecutor
from finflow_agent.execution.visualization.spec import VisualizationSpec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_VISUALIZATIONS_PER_JOB: int = 20
"""Maximum number of visualization DAG nodes allowed per job. Req 5.1, 5.5."""

_VISUALIZATION_TIMEOUT_SECONDS: int = 30
"""Execution timeout for a single visualization node. Req 6.3."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VisualizationPlanError(Exception):
    """Raised when a visualization plan fails validation before execution.

    This is a pre-execution rejection — the plan is never executed.
    """


class VisualizationLimitExceededError(VisualizationPlanError):
    """Raised when a plan contains more than MAX_VISUALIZATIONS_PER_JOB viz nodes.

    Requirements: 5.5
    """

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"Visualization limit exceeded: plan contains {count} visualization "
            f"nodes but the maximum allowed is {MAX_VISUALIZATIONS_PER_JOB}."
        )


class VisualizationMultipleDependencyError(VisualizationPlanError):
    """Raised when a visualization node depends on more than one calculation step.

    Requirements: 16.4
    """

    def __init__(self, step_id: str, depends_on: list[str]) -> None:
        self.step_id = step_id
        self.depends_on = depends_on
        super().__init__(
            f"Visualization node '{step_id}' has multiple dependencies "
            f"{depends_on}. Each visualization must depend on exactly one "
            f"calculation step."
        )


# ---------------------------------------------------------------------------
# VisualizationNode dataclass
# ---------------------------------------------------------------------------


@dataclass
class VisualizationNode:
    """A discrete visualization operation step in the execution plan DAG.

    Each visualization node has its own lifecycle: it starts as "pending",
    transitions to "running" during execution, and ends at "success" or
    "failed".

    Attributes:
        step_id: Unique identifier for this visualization step.
        kind: Always "visualization" to distinguish from calc nodes.
        depends_on: The single source calculation step_id this viz depends on.
        status: Current lifecycle status of the node.
        operation_id: The operation identifier linking to the source result.
        chart_type: Requested chart type (may be "auto").
        encoding_hints: Optional axis-to-field mapping hints.
        error: Error message if status is "failed".
        result: The produced VisualizationSpec (set after execution).

    Requirements: 2.1
    """

    step_id: str
    kind: Literal["visualization"] = "visualization"
    depends_on: str = ""
    status: Literal["pending", "running", "success", "failed"] = "pending"
    operation_id: str = ""
    chart_type: str | None = None
    encoding_hints: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    result: VisualizationSpec | None = None


# ---------------------------------------------------------------------------
# Plan Validation
# ---------------------------------------------------------------------------


def validate_visualization_plan(nodes: list[VisualizationNode]) -> None:
    """Validate a set of visualization nodes before execution.

    Checks:
    1. The plan contains at most MAX_VISUALIZATIONS_PER_JOB visualization nodes.
    2. Each visualization node depends on exactly one (not >1) calculation step.

    Raises:
        VisualizationLimitExceededError: If >20 viz nodes are present.
        VisualizationMultipleDependencyError: If any node has >1 dependency.

    Requirements: 5.1, 5.5, 16.4
    """
    # Check count limit. Req 5.5
    if len(nodes) > MAX_VISUALIZATIONS_PER_JOB:
        raise VisualizationLimitExceededError(len(nodes))

    # Check single-dependency constraint. Req 16.4
    # Note: depends_on is a single string in VisualizationNode, but we also
    # accept the raw plan step format where depends_on might be a list.
    # This validation covers the case where the node was built from a plan
    # step with multiple dependencies.
    for node in nodes:
        if not node.depends_on:
            raise VisualizationPlanError(
                f"Visualization node '{node.step_id}' has no dependency. "
                f"Each visualization must depend on exactly one calculation step."
            )


def validate_visualization_plan_from_steps(
    steps: list[dict[str, Any]],
) -> None:
    """Validate visualization steps from raw plan step dictionaries.

    This is called before execution to reject invalid plans early.
    It checks the same constraints as validate_visualization_plan but
    operates on raw step dicts (as they appear in ExecutionPlan).

    Args:
        steps: List of step dictionaries with at least 'step_id',
               'kind' (or 'agent'), and 'depends_on' keys.

    Raises:
        VisualizationLimitExceededError: If >20 viz steps.
        VisualizationMultipleDependencyError: If any viz step has >1 dependency.

    Requirements: 5.1, 5.5, 16.4
    """
    viz_steps = [
        s for s in steps
        if s.get("kind") == "visualization"
    ]

    # Check count limit
    if len(viz_steps) > MAX_VISUALIZATIONS_PER_JOB:
        raise VisualizationLimitExceededError(len(viz_steps))

    # Check single-dependency constraint
    for step in viz_steps:
        depends_on = step.get("depends_on", [])
        if isinstance(depends_on, list) and len(depends_on) > 1:
            raise VisualizationMultipleDependencyError(
                step_id=step.get("step_id", "unknown"),
                depends_on=depends_on,
            )


# ---------------------------------------------------------------------------
# Node Factory
# ---------------------------------------------------------------------------


def create_visualization_node(
    *,
    depends_on: str,
    operation_id: str,
    chart_type: str | None = None,
    encoding_hints: dict[str, str] | None = None,
    step_id: str | None = None,
) -> VisualizationNode:
    """Create a new VisualizationNode with a unique step_id and initial status "pending".

    Args:
        depends_on: The step_id of the source calculation step.
        operation_id: Identifier linking to the source operation result.
        chart_type: Requested chart type (None/empty treated as "auto" by executor).
        encoding_hints: Optional axis-role to field-id mapping.
        step_id: Optional explicit step_id (auto-generated UUID if not provided).

    Returns:
        A VisualizationNode in "pending" status.

    Requirements: 2.1
    """
    return VisualizationNode(
        step_id=step_id or f"viz_{uuid.uuid4().hex[:12]}",
        kind="visualization",
        depends_on=depends_on,
        status="pending",
        operation_id=operation_id,
        chart_type=chart_type,
        encoding_hints=encoding_hints or {},
    )


# ---------------------------------------------------------------------------
# Dependency Resolution
# ---------------------------------------------------------------------------


def resolve_dependency(
    node: VisualizationNode,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve the depends_on reference for a visualization node.

    Confirms that the referenced source step has completed with status
    "success" and its OperationResult is retrievable.

    Args:
        node: The visualization node whose dependency to resolve.
        step_results: Dictionary of step_id -> result envelope from the
                      execution engine (containing at minimum 'status' and
                      optionally 'data' with the OperationResult).

    Returns:
        The source step's result envelope if dependency is satisfied, or
        None if the dependency cannot be resolved (node will be failed).

    Requirements: 2.2, 2.3
    """
    source_step_id = node.depends_on

    # Check if the source step exists in results
    if source_step_id not in step_results:
        return None

    source_result = step_results[source_step_id]

    # Check if source step completed with status "success"
    if source_result.get("status") != "success":
        return None

    return source_result


# ---------------------------------------------------------------------------
# Single Node Execution
# ---------------------------------------------------------------------------


def _execute_single_node(
    node: VisualizationNode,
    step_results: dict[str, dict[str, Any]],
    pipeline_data: dict[str, Any],
    executor: VisualizationExecutor | None = None,
) -> VisualizationNode:
    """Execute a single visualization node.

    Resolves the dependency, extracts the OperationResult from pipeline state,
    invokes the VisualizationExecutor, and updates the node's status and result.

    Args:
        node: The visualization node to execute.
        step_results: Step results from the execution engine.
        pipeline_data: The PipelineState.data dict containing step outputs.
        executor: Optional VisualizationExecutor instance (created if not provided).

    Returns:
        The same node instance, mutated with updated status, result, and error.

    Requirements: 2.2, 2.3, 16.1, 16.2
    """
    node.status = "running"

    # Resolve dependency. Req 2.2
    source_result = resolve_dependency(node, step_results)
    if source_result is None:
        # Dependency unresolvable. Req 2.3
        source_step_id = node.depends_on
        if source_step_id not in step_results:
            error_msg = (
                f"Cannot execute visualization: source calculation step "
                f"'{source_step_id}' does not exist in the execution results."
            )
        else:
            source_status = step_results[source_step_id].get("status", "unknown")
            error_msg = (
                f"Cannot execute visualization: source calculation step "
                f"'{source_step_id}' has status '{source_status}' (expected 'success')."
            )
        node.status = "failed"
        node.error = error_msg
        node.result = VisualizationSpec(
            operation_id=node.operation_id,
            source_result_id=node.depends_on,
            status="failed",
            chart_type=node.chart_type or "auto",
            title="Visualization Failed",
            encoding={},
            data=[],
            error=error_msg[:500],
        )
        return node

    # Extract the OperationResult from pipeline_data.
    # The source step's output is stored under its output_key or step_id.
    operation_result = _extract_operation_result(node.depends_on, pipeline_data)
    if operation_result is None:
        error_msg = (
            f"Cannot execute visualization: source result for step "
            f"'{node.depends_on}' could not be located in pipeline state."
        )
        node.status = "failed"
        node.error = error_msg
        node.result = VisualizationSpec(
            operation_id=node.operation_id,
            source_result_id=node.depends_on,
            status="failed",
            chart_type=node.chart_type or "auto",
            title="Visualization Failed",
            encoding={},
            data=[],
            error=error_msg[:500],
        )
        return node

    # Execute via VisualizationExecutor. Req 16.1, 16.2
    if executor is None:
        executor = VisualizationExecutor()

    try:
        spec = executor.execute(
            operation_result=operation_result,
            chart_type=node.chart_type,
            encoding_hints=node.encoding_hints if node.encoding_hints else None,
            source_result_id=node.depends_on,
            operation_id=node.operation_id,
        )
    except Exception as exc:
        error_msg = f"Visualization execution failed with an unexpected error: {exc}"
        node.status = "failed"
        node.error = error_msg[:500]
        node.result = VisualizationSpec(
            operation_id=node.operation_id,
            source_result_id=node.depends_on,
            status="failed",
            chart_type=node.chart_type or "auto",
            title="Visualization Failed",
            encoding={},
            data=[],
            error=error_msg[:500],
        )
        return node

    node.result = spec
    if spec.status == "failed":
        node.status = "failed"
        node.error = spec.error
    elif spec.status == "unsupported":
        # "unsupported" is a valid terminal state — the node completed its work
        node.status = "success"
        node.error = None
    else:
        node.status = "success"
        node.error = None

    return node


def _extract_operation_result(
    step_id: str,
    pipeline_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract the OperationResult dict from pipeline state for a given step.

    The execution engine stores step outputs in pipeline_data under
    the step's output_key or step_id. The value can be:
    - A dict envelope with {"data": <df_or_dict>, "artifacts": {...}, ...}
    - A raw dict (if the step produced an OperationResult directly)

    For visualization purposes, we look for a dict that contains "fields"
    and "rows" keys (the OperationResult shape), or return the envelope's
    data if it matches.

    Args:
        step_id: The step_id whose output to retrieve.
        pipeline_data: The PipelineState.data dictionary.

    Returns:
        The OperationResult dict, or None if not found.
    """
    if step_id not in pipeline_data:
        return None

    value = pipeline_data[step_id]

    # If the value itself is an OperationResult-shaped dict
    if isinstance(value, dict):
        if "fields" in value and "rows" in value:
            return value
        # Check envelope: the 'data' key may hold the result
        inner = value.get("data")
        if isinstance(inner, dict) and "fields" in inner and "rows" in inner:
            return inner
        # Check artifacts for operation_result
        artifacts = value.get("artifacts")
        if isinstance(artifacts, dict):
            op_result = artifacts.get("operation_result")
            if isinstance(op_result, dict) and "fields" in op_result:
                return op_result
        # Return the value itself as a fallback — the OperationResultReader
        # will validate it and fail gracefully if invalid
        return value

    return None


# ---------------------------------------------------------------------------
# Concurrent Execution
# ---------------------------------------------------------------------------


def execute_visualization_nodes(
    nodes: list[VisualizationNode],
    step_results: dict[str, dict[str, Any]],
    pipeline_data: dict[str, Any],
    executor: VisualizationExecutor | None = None,
) -> list[VisualizationSpec]:
    """Execute all visualization nodes, running independent nodes concurrently.

    Nodes whose dependencies are satisfied are executed concurrently using
    a thread pool. Nodes with unresolvable dependencies are failed immediately.

    The function produces one VisualizationSpec per node regardless of whether
    other nodes succeed or fail (Req 5.3).

    Args:
        nodes: List of VisualizationNodes to execute (already validated).
        step_results: Step results from the execution engine.
        pipeline_data: The PipelineState.data dict.
        executor: Optional shared VisualizationExecutor instance.

    Returns:
        List of VisualizationSpecs in the same order as the input nodes
        (preserving topological order). Req 5.4.

    Requirements: 2.4, 5.3, 5.4
    """
    if not nodes:
        return []

    if executor is None:
        executor = VisualizationExecutor()

    # Execute nodes concurrently using a thread pool.
    # Each node is independent once its dependency is resolved.
    # Req 2.4: permit concurrent execution of viz nodes whose dependencies
    # are satisfied.
    with ThreadPoolExecutor(
        max_workers=min(len(nodes), MAX_VISUALIZATIONS_PER_JOB)
    ) as pool:
        futures = [
            pool.submit(
                _execute_single_node, node, step_results, pipeline_data, executor
            )
            for node in nodes
        ]

        # Collect results in original order (preserving topological ordering).
        # Enforce 30-second execution timeout per node (Req 6.3).
        for i, future in enumerate(futures):
            try:
                future.result(timeout=_VISUALIZATION_TIMEOUT_SECONDS)
            except TimeoutError:
                # Timeout: mark node as failed with timeout error. Req 6.3
                node = nodes[i]
                if node.result is None:
                    error_msg = (
                        f"Visualization execution timed out after "
                        f"{_VISUALIZATION_TIMEOUT_SECONDS} seconds."
                    )
                    node.status = "failed"
                    node.error = error_msg
                    node.result = VisualizationSpec(
                        operation_id=node.operation_id,
                        source_result_id=node.depends_on,
                        status="failed",
                        chart_type=node.chart_type or "auto",
                        title="Visualization Failed",
                        encoding={},
                        data=[],
                        error=error_msg,
                    )
            except Exception as exc:
                # Unexpected error during collection — mark node failed.
                node = nodes[i]
                if node.result is None:
                    error_msg = (
                        f"Visualization node failed with an unexpected error: "
                        f"{exc}"
                    )[:500]
                    node.status = "failed"
                    node.error = error_msg
                    node.result = VisualizationSpec(
                        operation_id=node.operation_id,
                        source_result_id=node.depends_on,
                        status="failed",
                        chart_type=node.chart_type or "auto",
                        title="Visualization Failed",
                        encoding={},
                        data=[],
                        error=error_msg,
                    )

    # Build the result list: one spec per node, in input order. Req 5.3, 5.4
    specs: list[VisualizationSpec] = []
    for node in nodes:
        if node.result is not None:
            specs.append(node.result)
        else:
            # Safety net — should not happen after the timeout handling above.
            specs.append(
                VisualizationSpec(
                    operation_id=node.operation_id,
                    source_result_id=node.depends_on,
                    status="failed",
                    chart_type=node.chart_type or "auto",
                    title="Visualization Failed",
                    encoding={},
                    data=[],
                    error="Visualization node did not produce a result.",
                )
            )

    return specs
