"""Execution package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ColumnNotInPackageError",
    "ContentHashMismatchError",
    "ExecutionEngine",
    "ExecutionError",
    "ExecutionResult",
    "Executor",
    "ExecutorIntentPackage",
    # Visualization runner exports
    "MAX_VISUALIZATIONS_PER_JOB",
    "VisualizationLimitExceededError",
    "VisualizationMultipleDependencyError",
    "VisualizationNode",
    "VisualizationPlanError",
    "create_visualization_node",
    "execute_visualization_nodes",
    "validate_visualization_plan",
    "validate_visualization_plan_from_steps",
    # Visualization job status exports
    "JobRunner",
    "VisualizationJobResult",
    "VisualizationJobStatusHandler",
]

# Names that live in the visualization_runner module
_VIZ_RUNNER_NAMES = {
    "MAX_VISUALIZATIONS_PER_JOB",
    "VisualizationLimitExceededError",
    "VisualizationMultipleDependencyError",
    "VisualizationNode",
    "VisualizationPlanError",
    "create_visualization_node",
    "execute_visualization_nodes",
    "validate_visualization_plan",
    "validate_visualization_plan_from_steps",
}

# Names that live in the visualization_job_status module
_VIZ_JOB_STATUS_NAMES = {
    "VisualizationJobResult",
    "VisualizationJobStatusHandler",
}

# Names that live in the job_runner module
_JOB_RUNNER_NAMES = {
    "JobRunner",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _VIZ_RUNNER_NAMES:
        from finflow_agent.execution import visualization_runner as _viz_runner

        value = getattr(_viz_runner, name)
    elif name in _VIZ_JOB_STATUS_NAMES:
        from finflow_agent.execution import visualization_job_status as _viz_status

        value = getattr(_viz_status, name)
    elif name in _JOB_RUNNER_NAMES:
        from finflow_agent.execution import job_runner as _jr

        value = getattr(_jr, name)
    else:
        from finflow_agent.execution import engine as _engine

        value = getattr(_engine, name)

    globals()[name] = value
    return value
