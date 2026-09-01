"""VisualizationSpec model and supporting types for chart visualization.

Defines the versioned contract shared between backend persistence and frontend
rendering. The VisualizationSpec captures all information needed to render a
chart without performing any business calculations.

Key types:
- DataShape: enum classifying the shape of operation result data
- FieldMetadata: structured metadata for each field in an OperationResult
- VisualizationSpec: the versioned JSON contract for chart rendering

Requirements: 9.1, 9.2, 14.1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DataShape(str, Enum):
    """Classification of the data shape in an OperationResult.

    Used to determine appropriate chart type during auto-selection and
    to validate chart compatibility.

    Requirements: 14.1
    """

    TIME_SERIES = "time_series"
    CATEGORICAL_SERIES = "categorical_series"
    HISTOGRAM_BINS = "histogram_bins"
    SCATTER_POINTS = "scatter_points"
    SCALAR = "scalar"


@dataclass(frozen=True)
class FieldMetadata:
    """Structured metadata for each field in an OperationResult.

    Attributes:
        id: Stable key identifying this field.
        label: Human-readable display name.
        data_type: The data type of the field values.
        role: The semantic role of this field in the dataset.
        unit: Optional unit of measure (e.g., "USD", "kg").
        aggregation: Optional aggregation applied (e.g., "sum", "avg").

    Requirements: 14.1
    """

    id: str
    label: str
    data_type: Literal["string", "integer", "float", "datetime"]
    role: Literal["category", "measure", "time", "dimension", "identifier"]
    unit: str | None = None
    aggregation: str | None = None


class VisualizationSpec(BaseModel):
    """Versioned contract for chart visualization data.

    Produced by the Visualization_Executor and consumed by both the backend
    persistence layer and the frontend VisualizationRenderer component.

    Fields:
        schema_version: Contract version string (initially "1.0").
        visualization_id: Unique UUID identifying this visualization.
        operation_id: Identifier of the operation that produced this visualization.
        source_result_id: Identifier of the source OperationResult consumed.
        status: Lifecycle status — "ready", "unsupported", or "failed".
        chart_type: The resolved chart type (never "auto" in output).
        title: Human-readable chart title (max 200 characters).
        encoding: Mapping of axis roles to field IDs.
        data: Array of row objects for chart rendering.
        options: Chart-specific configuration settings.
        warnings: Informational messages (max 20 entries).
        error: Error description when status is "unsupported" or "failed".

    Requirements: 9.1, 9.2
    """

    model_config = ConfigDict(strict=True)

    schema_version: str = "1.0"
    visualization_id: str = Field(default_factory=lambda: str(uuid4()))
    operation_id: str
    source_result_id: str
    status: Literal["ready", "unsupported", "failed"]
    chart_type: str
    title: str = Field(max_length=200)
    encoding: dict[str, str] = Field(default_factory=dict)
    data: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    error: str | None = None
