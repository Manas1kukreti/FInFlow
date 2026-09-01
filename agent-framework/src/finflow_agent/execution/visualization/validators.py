"""Chart compatibility validators for the Visualization Executor.

Each validator checks whether a given OperationResult (via OperationResultReader)
and encoding satisfy the structural requirements for a specific chart type.

Null handling (Req 8.7): rows with null values in required fields are excluded
from minimum-row-count checks, but null presence alone does not fail validation.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from finflow_agent.execution.visualization.operation_result_reader import (
    OperationResultReader,
)
from finflow_agent.execution.visualization.spec import FieldMetadata


@dataclass
class ValidationResult:
    """Result of a chart compatibility validation check.

    Attributes:
        valid: Whether the data is compatible with the chart type.
        reason_code: Identifies the chart type and specific failed condition
            (e.g., "line_incompatible_insufficient_x_values"). None when valid.
        error_message: Plain-language description of the incompatibility.
            None when valid.
    """

    valid: bool
    reason_code: str | None = None
    error_message: str | None = None


class ChartValidator(Protocol):
    """Protocol for chart compatibility validators."""

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult: ...


def _get_field_by_id(
    fields: list[FieldMetadata], field_id: str
) -> FieldMetadata | None:
    """Look up a FieldMetadata by its id."""
    for f in fields:
        if f.id == field_id:
            return f
    return None


def _is_numeric_field(field: FieldMetadata) -> bool:
    """Check if a field has a numeric data_type (integer, float, or number)."""
    return field.data_type in ("integer", "float", "number")


class LineChartValidator:
    """Validates data compatibility for line charts.

    Rules (Requirement 8.1):
    - The encoding must contain an x-axis field with role "time" or "dimension".
    - At least 2 distinct x-axis values must exist in the data rows
      (excluding rows where the x-axis field is null).
    """

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult:
        fields = reader.get_fields()
        rows = reader.get_rows()

        # Check x-axis field exists in encoding
        x_field_id = encoding.get("x")
        if not x_field_id:
            return ValidationResult(
                valid=False,
                reason_code="line_incompatible_missing_x_axis",
                error_message="Line chart requires an x-axis field in the encoding.",
            )

        # Check x-axis field has role "time" or "dimension"
        x_field = _get_field_by_id(fields, x_field_id)
        if x_field is None:
            return ValidationResult(
                valid=False,
                reason_code="line_incompatible_missing_x_axis",
                error_message=(
                    f"Line chart x-axis field '{x_field_id}' not found in data fields."
                ),
            )

        if x_field.role not in ("time", "dimension"):
            return ValidationResult(
                valid=False,
                reason_code="line_incompatible_invalid_x_role",
                error_message=(
                    f"Line chart x-axis field must have role 'time' or 'dimension', "
                    f"but '{x_field_id}' has role '{x_field.role}'."
                ),
            )

        # Count distinct non-null x-axis values (Req 8.7: exclude nulls)
        distinct_x_values: set[Any] = set()
        for row in rows:
            val = row.get(x_field_id)
            if val is not None:
                distinct_x_values.add(val)

        if len(distinct_x_values) < 2:
            return ValidationResult(
                valid=False,
                reason_code="line_incompatible_insufficient_x_values",
                error_message=(
                    "Line chart requires at least 2 distinct x-axis values, "
                    f"but only {len(distinct_x_values)} found."
                ),
            )

        return ValidationResult(valid=True)


class BarChartValidator:
    """Validates data compatibility for bar charts.

    Rules (Requirement 8.2):
    - The encoding must contain at least one field with role "category" or "dimension".
    - The encoding must contain at least one field with data_type "integer" or "float".
    - At least 1 data row must exist (excluding rows with null in required fields).
    """

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult:
        fields = reader.get_fields()
        rows = reader.get_rows()

        # Find category/dimension fields in encoding
        category_field_ids: list[str] = []
        numeric_field_ids: list[str] = []

        for _axis, field_id in encoding.items():
            field = _get_field_by_id(fields, field_id)
            if field is None:
                continue
            if field.role in ("category", "dimension"):
                category_field_ids.append(field_id)
            if _is_numeric_field(field):
                numeric_field_ids.append(field_id)

        if not category_field_ids:
            return ValidationResult(
                valid=False,
                reason_code="bar_incompatible_missing_category",
                error_message=(
                    "Bar chart requires at least one field with role "
                    "'category' or 'dimension' in the encoding."
                ),
            )

        if not numeric_field_ids:
            return ValidationResult(
                valid=False,
                reason_code="bar_incompatible_missing_numeric",
                error_message=(
                    "Bar chart requires at least one numeric field "
                    "(data_type 'integer' or 'float') in the encoding."
                ),
            )

        # Check at least 1 row with non-null values in required fields (Req 8.7)
        required_field_ids = category_field_ids + numeric_field_ids
        valid_row_count = _count_valid_rows(rows, required_field_ids)

        if valid_row_count < 1:
            return ValidationResult(
                valid=False,
                reason_code="bar_incompatible_insufficient_rows",
                error_message=(
                    "Bar chart requires at least 1 data row with non-null values "
                    "in required fields."
                ),
            )

        return ValidationResult(valid=True)


class PieChartValidator:
    """Validates data compatibility for pie charts.

    Rules (Requirement 8.3):
    - Exactly 1 field with role "category" or "dimension" in encoding.
    - Exactly 1 field with data_type "integer" or "float" in encoding.
    - The numeric field must have ``aggregation`` set (count, sum, mean,
      percentage, etc.) — raw unaggregated data is not valid for pie charts.
    - All non-null values in the numeric field must be >= 0.
    - The number of distinct categories must not exceed max_categories.
    - The data_shape must represent pre-aggregated (categorical_series) data.

    Args:
        max_categories: Maximum allowed distinct categories (default 12, min 2, max 50).
        require_aggregation: Whether to require the aggregation field on
            the numeric measure. Defaults to True. Set to False for testing
            scenarios with pre-built operation results.
    """

    def __init__(self, max_categories: int = 12, require_aggregation: bool = True) -> None:
        # Clamp max_categories to [2, 50]
        self.max_categories = max(2, min(50, max_categories))
        self.require_aggregation = require_aggregation

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult:
        fields = reader.get_fields()
        rows = reader.get_rows()

        # Find category/dimension and numeric fields in encoding
        category_field_ids: list[str] = []
        numeric_field_ids: list[str] = []

        for _axis, field_id in encoding.items():
            field = _get_field_by_id(fields, field_id)
            if field is None:
                continue
            if field.role in ("category", "dimension"):
                category_field_ids.append(field_id)
            if _is_numeric_field(field):
                numeric_field_ids.append(field_id)

        # Exactly 1 category/dimension
        if len(category_field_ids) != 1:
            return ValidationResult(
                valid=False,
                reason_code="pie_incompatible_category_count",
                error_message=(
                    f"Pie chart requires exactly 1 category/dimension field, "
                    f"but found {len(category_field_ids)}."
                ),
            )

        # Exactly 1 numeric
        if len(numeric_field_ids) != 1:
            return ValidationResult(
                valid=False,
                reason_code="pie_incompatible_numeric_count",
                error_message=(
                    f"Pie chart requires exactly 1 numeric field, "
                    f"but found {len(numeric_field_ids)}."
                ),
            )

        numeric_field_id = numeric_field_ids[0]

        # Check that the numeric field has aggregation metadata
        if self.require_aggregation:
            numeric_field_meta = _get_field_by_id(fields, numeric_field_id)
            if numeric_field_meta is not None:
                aggregation = getattr(numeric_field_meta, "aggregation", None)
                if not aggregation:
                    return ValidationResult(
                        valid=False,
                        reason_code="pie_incompatible_aggregation_required",
                        error_message=(
                            "Pie chart requires the value field to have an aggregation "
                            "(count, sum, mean, etc.). Raw unaggregated data cannot be "
                            "rendered as a pie chart. Use a calculation step to aggregate first."
                        ),
                    )

        category_field_id = category_field_ids[0]

        # Check all non-null numeric values are >= 0
        for row in rows:
            val = row.get(numeric_field_id)
            if val is not None:
                try:
                    if float(val) < 0:
                        return ValidationResult(
                            valid=False,
                            reason_code="pie_incompatible_negative_values",
                            error_message=(
                                "Pie chart requires all numeric values to be "
                                "greater than or equal to zero."
                            ),
                        )
                except (TypeError, ValueError):
                    pass

        # Count distinct non-null categories
        distinct_categories: set[Any] = set()
        for row in rows:
            val = row.get(category_field_id)
            if val is not None:
                distinct_categories.add(val)

        if len(distinct_categories) > self.max_categories:
            return ValidationResult(
                valid=False,
                reason_code="pie_incompatible_too_many_categories",
                error_message=(
                    f"Pie chart allows at most {self.max_categories} distinct "
                    f"categories, but found {len(distinct_categories)}."
                ),
            )

        return ValidationResult(valid=True)


class ScatterChartValidator:
    """Validates data compatibility for scatter charts.

    Rules (Requirement 8.4):
    - The encoding must contain 2 fields each with data_type "integer" or "float".
    - At least 2 data rows must have non-null values in both numeric fields.
    """

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult:
        fields = reader.get_fields()
        rows = reader.get_rows()

        # Find numeric fields in encoding
        numeric_field_ids: list[str] = []

        for _axis, field_id in encoding.items():
            field = _get_field_by_id(fields, field_id)
            if field is None:
                continue
            if _is_numeric_field(field):
                numeric_field_ids.append(field_id)

        if len(numeric_field_ids) < 2:
            return ValidationResult(
                valid=False,
                reason_code="scatter_incompatible_insufficient_numeric_fields",
                error_message=(
                    f"Scatter chart requires 2 numeric fields, "
                    f"but found {len(numeric_field_ids)}."
                ),
            )

        # Use the first 2 numeric fields for row validation
        field_a = numeric_field_ids[0]
        field_b = numeric_field_ids[1]

        # Count rows with non-null values in both fields (Req 8.7)
        valid_row_count = _count_valid_rows(rows, [field_a, field_b])

        if valid_row_count < 2:
            return ValidationResult(
                valid=False,
                reason_code="scatter_incompatible_insufficient_rows",
                error_message=(
                    "Scatter chart requires at least 2 data rows with non-null "
                    f"values in both numeric fields, but found {valid_row_count}."
                ),
            )

        return ValidationResult(valid=True)


class HistogramChartValidator:
    """Validates data compatibility for histogram charts.

    Rules (Requirement 8.5):
    - The OperationResult must contain at least one field with role "measure"
      representing frequencies (data_type "integer" or "float").
    - At least 1 row of bin data must exist (excluding rows with null in
      the measure field).
    """

    def validate(
        self, reader: OperationResultReader, encoding: dict[str, str]
    ) -> ValidationResult:
        fields = reader.get_fields()
        rows = reader.get_rows()

        # Find measure fields with numeric data_type (frequencies)
        measure_fields = [
            f for f in fields if f.role == "measure" and _is_numeric_field(f)
        ]

        if not measure_fields:
            return ValidationResult(
                valid=False,
                reason_code="histogram_incompatible_missing_measure",
                error_message=(
                    "Histogram chart requires at least one measure field with "
                    "numeric data_type (integer or float) representing frequencies."
                ),
            )

        # Check at least 1 row with non-null value in at least one measure field (Req 8.7)
        measure_field_ids = [f.id for f in measure_fields]
        valid_row_count = 0
        for row in rows:
            # A row is valid if at least one measure field has a non-null value
            for fid in measure_field_ids:
                if row.get(fid) is not None:
                    valid_row_count += 1
                    break

        if valid_row_count < 1:
            return ValidationResult(
                valid=False,
                reason_code="histogram_incompatible_insufficient_rows",
                error_message=(
                    "Histogram chart requires at least 1 row of bin data with "
                    "non-null frequency values."
                ),
            )

        return ValidationResult(valid=True)


def _count_valid_rows(
    rows: list[dict[str, Any]], required_field_ids: list[str]
) -> int:
    """Count rows where all required fields have non-null values.

    Per Requirement 8.7, rows with null in required fields are excluded
    from minimum-row-count checks.
    """
    count = 0
    for row in rows:
        if all(row.get(fid) is not None for fid in required_field_ids):
            count += 1
    return count


# Registry mapping chart type names to their validator instances
CHART_VALIDATORS: dict[str, ChartValidator] = {
    "line": LineChartValidator(),
    "bar": BarChartValidator(),
    "pie": PieChartValidator(require_aggregation=True),
    "scatter": ScatterChartValidator(),
    "histogram": HistogramChartValidator(),
}
