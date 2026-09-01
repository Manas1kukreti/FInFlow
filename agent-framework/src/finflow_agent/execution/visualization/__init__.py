"""Visualization subpackage for chart generation from operation results."""

from __future__ import annotations

from finflow_agent.execution.visualization.executor import VisualizationExecutor
from finflow_agent.execution.visualization.operation_result_reader import (
    OperationResultReader,
    OperationResultReaderError,
)
from finflow_agent.execution.visualization.spec import (
    DataShape,
    FieldMetadata,
    VisualizationSpec,
)

__all__ = [
    "DataShape",
    "FieldMetadata",
    "OperationResultReader",
    "OperationResultReaderError",
    "VisualizationExecutor",
    "VisualizationSpec",
]
