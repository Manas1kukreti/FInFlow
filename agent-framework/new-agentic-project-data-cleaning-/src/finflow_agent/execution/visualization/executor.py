"""VisualizationExecutor — produces a VisualizationSpec from an OperationResult.

The executor performs zero business calculations. It reads field metadata and
rows from the OperationResult via the OperationResultReader adapter, validates
field references and chart compatibility, selects a chart type (if "auto"),
and builds a VisualizationSpec with verbatim row data.

Requirements: 3.1, 3.2, 3.3, 3.4, 7.1, 7.2, 7.3, 7.4, 7.5, 9.3, 9.4, 9.5,
              15.1, 15.2, 15.3, 15.4, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6,
              18.1, 18.2, 18.4
"""

from __future__ import annotations

from typing import Any

from finflow_agent.execution.visualization.operation_result_reader import (
    OperationResultReader,
    OperationResultReaderError,
)
from finflow_agent.execution.visualization.spec import (
    DataShape,
    FieldMetadata,
    VisualizationSpec,
)
from finflow_agent.execution.visualization.capabilities import VISUALIZATION_CAPABILITIES
from finflow_agent.execution.visualization.validators import (
    CHART_VALIDATORS,
    ValidationResult,
)


# Supported chart types (case-sensitive lowercase only). Req 15.1, 15.2
SUPPORTED_CHART_TYPES: set[str] = {"auto", "bar", "line", "pie", "scatter", "histogram"}

# Deterministic mapping from DataShape to chart type for auto-selection. Req 7.1
_AUTO_CHART_MAP: dict[DataShape, str] = {
    DataShape.TIME_SERIES: "line",
    DataShape.CATEGORICAL_SERIES: "bar",
    DataShape.HISTOGRAM_BINS: "histogram",
    DataShape.SCATTER_POINTS: "scatter",
}

_MAX_TITLE_LENGTH: int = 200
_MAX_ERROR_LENGTH: int = 500


class VisualizationExecutor:
    """Produces a VisualizationSpec from an OperationResult without calculations.

    The executor's pipeline:
    1. Normalize chart_type (None/empty → "auto")
    2. Validate chart_type against the supported set
    3. Read field metadata and rows via OperationResultReader
    4. Validate field references in encoding (existence, data_type, role)
    5. Run chart compatibility validation
    6. Auto-select chart type if "auto"
    7. Build the VisualizationSpec with verbatim row data

    Requirements: 3.1, 3.2, 3.3, 3.4, 7.1-7.5, 9.3-9.5, 15.1-15.4,
                  17.1-17.6, 18.1, 18.2, 18.4
    """

    SUPPORTED_CHART_TYPES = SUPPORTED_CHART_TYPES

    def execute(
        self,
        operation_result: dict[str, Any],
        chart_type: str | None,
        encoding_hints: dict[str, str] | None,
        source_result_id: str,
        operation_id: str,
    ) -> VisualizationSpec:
        """Execute the visualization pipeline and produce a VisualizationSpec.

        Args:
            operation_result: The raw OperationResult dictionary with fields and rows.
            chart_type: Requested chart type (None/empty treated as "auto").
            encoding_hints: Mapping of axis roles to field IDs (e.g., {"x": "month", "y": "revenue"}).
            source_result_id: Identifier of the source OperationResult.
            operation_id: Identifier of the operation producing this visualization.

        Returns:
            A fully-populated VisualizationSpec instance.
        """
        try:
            return self._execute_internal(
                operation_result, chart_type, encoding_hints, source_result_id, operation_id
            )
        except Exception as exc:
            # Catch-all for unhandled exceptions. Req 9.4, 18.1
            error_msg = self._format_error(
                f"An unexpected error occurred while generating the visualization: {exc}"
            )
            return VisualizationSpec(
                operation_id=operation_id,
                source_result_id=source_result_id,
                status="failed",
                chart_type=chart_type if chart_type else "auto",
                title=self._generate_title(chart_type or "auto", None),
                encoding={},
                data=[],
                error=error_msg,
            )

    def _execute_internal(
        self,
        operation_result: dict[str, Any],
        chart_type: str | None,
        encoding_hints: dict[str, str] | None,
        source_result_id: str,
        operation_id: str,
    ) -> VisualizationSpec:
        """Internal execution logic, separated for clean exception handling."""

        # Step 1: Normalize chart_type. Req 15.4
        normalized_chart_type = self._normalize_chart_type(chart_type)

        # Step 2: Validate chart_type against supported set. Req 15.1, 15.2, 15.3
        if normalized_chart_type not in SUPPORTED_CHART_TYPES:
            return VisualizationSpec(
                operation_id=operation_id,
                source_result_id=source_result_id,
                status="unsupported",
                chart_type=normalized_chart_type,
                title=self._generate_title(normalized_chart_type, None),
                encoding={},
                data=[],
                error=self._format_error(
                    f"Chart type '{normalized_chart_type}' is not supported. "
                    f"Supported types are: bar, line, pie, scatter, histogram."
                ),
            )

        # Step 3: Read via OperationResultReader adapter. Req 14.2, 14.3
        try:
            reader = OperationResultReader(operation_result)
        except OperationResultReaderError as exc:
            return VisualizationSpec(
                operation_id=operation_id,
                source_result_id=source_result_id,
                status="failed",
                chart_type=normalized_chart_type,
                title=self._generate_title(normalized_chart_type, None),
                encoding={},
                data=[],
                error=self._format_error(
                    f"The source data is invalid and cannot be visualized: {exc}"
                ),
            )

        fields = reader.get_fields()
        rows = reader.get_rows()
        data_shape = reader.get_data_shape()

        # Resolve encoding (use hints or empty)
        encoding = encoding_hints if encoding_hints else {}

        # Step 4: Validate field references in encoding. Req 17.1-17.6
        field_validation_error = self._validate_field_references(encoding, fields)
        if field_validation_error is not None:
            return VisualizationSpec(
                operation_id=operation_id,
                source_result_id=source_result_id,
                status="unsupported",
                chart_type=normalized_chart_type,
                title=self._generate_title(normalized_chart_type, self._find_primary_measure(encoding, fields)),
                encoding={},
                data=[],
                error=self._format_error(field_validation_error),
            )

        # Step 5: Resolve auto chart type. Req 7.1-7.4
        resolved_chart_type = self._resolve_chart_type(normalized_chart_type, data_shape)

        # Step 6: Run chart compatibility validation. Req 8.1-8.7
        if encoding.get("series") and not VISUALIZATION_CAPABILITIES.get(resolved_chart_type, VISUALIZATION_CAPABILITIES["auto"]).supports_series:
            return VisualizationSpec(
                operation_id=operation_id,
                source_result_id=source_result_id,
                status="unsupported",
                chart_type=resolved_chart_type,
                title=self._generate_title(resolved_chart_type, self._find_primary_measure(encoding, fields)),
                encoding={},
                data=[],
                error=self._format_error(
                    f"{resolved_chart_type.capitalize()} chart does not support a secondary series field."
                ),
            )
        validator = CHART_VALIDATORS.get(resolved_chart_type)
        if validator is not None:
            validation_result: ValidationResult = validator.validate(reader, encoding)
            if not validation_result.valid:
                return VisualizationSpec(
                    operation_id=operation_id,
                    source_result_id=source_result_id,
                    status="unsupported",
                    chart_type=resolved_chart_type,
                    title=self._generate_title(resolved_chart_type, self._find_primary_measure(encoding, fields)),
                    encoding={},
                    data=[],
                    error=self._format_error(
                        validation_result.error_message or "Chart data is incompatible with the requested chart type."
                    ),
                )

        # Step 7: Build VisualizationSpec with mapped data. Req 3.1, 3.2, 3.4
        primary_measure = self._find_primary_measure(encoding, fields)
        title = self._generate_title(resolved_chart_type, primary_measure)

        # Map encoding keys to chart-type-specific keys expected by frontend
        frontend_encoding = self._map_encoding_for_frontend(encoding, resolved_chart_type, fields)

        return VisualizationSpec(
            operation_id=operation_id,
            source_result_id=source_result_id,
            status="ready",
            chart_type=resolved_chart_type,
            title=title,
            encoding=frontend_encoding,
            data=rows,  # Zero-calculation: copy rows verbatim. Req 3.1, 3.4
        )

    def _map_encoding_for_frontend(
        self, encoding: dict[str, str], chart_type: str, fields: list[FieldMetadata]
    ) -> dict[str, str]:
        """Map generic x/y encoding to chart-type-specific keys expected by frontend.

        Frontend components expect:
        - Pie: {category, value, category_label, value_label}
        - Bar: {x, y, x_label, y_label}
        - Line: {x, y, x_label, y_label}
        - Scatter: {x, y, x_label, y_label}
        """
        if not encoding:
            # Auto-detect from fields
            category_fields = [f for f in fields if f.role in ("category", "dimension")]
            measure_fields = [f for f in fields if f.role == "measure"]
            if chart_type == "pie" and category_fields and measure_fields:
                return {
                    "category": category_fields[0].id,
                    "value": measure_fields[0].id,
                    "category_label": category_fields[0].label,
                    "value_label": measure_fields[0].label,
                }
            elif category_fields and measure_fields:
                return {
                    "x": category_fields[0].id,
                    "y": measure_fields[0].id,
                    "x_label": category_fields[0].label,
                    "y_label": measure_fields[0].label,
                }
            return encoding

        x_field_id = encoding.get("x", "")
        y_field_id = encoding.get("y", "")
        series_field_id = encoding.get("series", "")
        field_map = {f.id: f for f in fields}
        x_label = field_map[x_field_id].label if x_field_id in field_map else x_field_id
        y_label = field_map[y_field_id].label if y_field_id in field_map else y_field_id
        series_label = field_map[series_field_id].label if series_field_id in field_map else series_field_id

        if chart_type == "pie":
            return {
                "category": x_field_id,
                "value": y_field_id,
                "category_label": x_label,
                "value_label": y_label,
            }

        # For bar, line, scatter — keep x/y but add labels
        return {
            **encoding,
            "x_label": x_label,
            "y_label": y_label,
            **({"series_label": series_label} if series_field_id else {}),
        }

    def _normalize_chart_type(self, chart_type: str | None) -> str:
        """Normalize null/empty/missing chart_type to "auto". Req 15.4."""
        if chart_type is None or chart_type == "":
            return "auto"
        return chart_type

    def _resolve_chart_type(self, chart_type: str, data_shape: DataShape) -> str:
        """Resolve "auto" chart type to a concrete type based on data_shape.

        Mapping (Req 7.1):
        - time_series → "line"
        - categorical_series → "bar"
        - histogram_bins → "histogram"
        - scatter_points → "scatter"
        - no match / scalar → "bar" (default, Req 7.3)

        Never selects "pie" (Req 7.2).
        Records resolved type (not "auto") in output (Req 7.4).
        """
        if chart_type != "auto":
            return chart_type

        return _AUTO_CHART_MAP.get(data_shape, "bar")

    def _validate_field_references(
        self, encoding: dict[str, str], fields: list[FieldMetadata]
    ) -> str | None:
        """Validate that field references in encoding are valid.

        Checks (Req 17.1-17.5):
        1. Every field ID in encoding exists in the source field metadata.
        2. Fields on measure axis (y, value) have data_type "integer" or "float".
        3. Fields on time axis (x when role is time) have data_type "datetime" or role "time".
        4. Fields on category axis (x when role is category) have role "category" or "dimension".

        Returns None if valid, or an error message string if invalid.
        Req 17.6: All field validations run before chart compatibility validation.
        """
        if not encoding:
            return None

        field_map: dict[str, FieldMetadata] = {f.id: f for f in fields}
        failing_fields: list[str] = []

        # Define which axis roles map to which validation rules
        measure_axes = {"y", "value", "size"}
        time_axes = {"x_time", "time"}
        category_axes = {"x_category", "category", "label", "name", "series"}

        for axis_role, field_id in encoding.items():
            # Check existence. Req 17.1
            field = field_map.get(field_id)
            if field is None:
                failing_fields.append(field_id)
                continue

            # Check data_type for measure axes. Req 17.2
            if axis_role in measure_axes:
                if field.data_type not in ("integer", "float", "number"):
                    failing_fields.append(field_id)
                    continue

            # Check time axis. Req 17.3
            if axis_role in time_axes:
                if field.data_type != "datetime" and field.role != "time":
                    failing_fields.append(field_id)
                    continue

            # Check category axis. Req 17.4
            if axis_role in category_axes:
                if field.role not in ("category", "dimension"):
                    failing_fields.append(field_id)
                    continue

            # For the generic "x" axis, apply role-based validation
            if axis_role == "x":
                # If the field has role "time", validate as time axis
                if field.role == "time":
                    if field.data_type != "datetime" and field.role != "time":
                        failing_fields.append(field_id)
                        continue
                # If field has role "measure", validate as measure axis
                elif field.role == "measure":
                    if field.data_type not in ("integer", "float", "number"):
                        failing_fields.append(field_id)
                        continue
                # Otherwise (category/dimension), validate as category
                elif field.role in ("category", "dimension"):
                    pass  # valid for category axis
                # If role doesn't match any expected pattern, it's still valid
                # (no additional constraint for generic x-axis fields)

        if failing_fields:
            field_list = ", ".join(f"'{fid}'" for fid in failing_fields)
            return (
                f"The following field references are invalid: {field_list}. "
                f"Please verify that the fields exist in the data and have "
                f"compatible data types for their assigned axis roles."
            )

        return None

    def _find_primary_measure(
        self, encoding: dict[str, str], fields: list[FieldMetadata]
    ) -> FieldMetadata | None:
        """Find the primary measure field from encoding for title generation.

        Looks for fields mapped to measure axis roles (y, value) first,
        then falls back to any numeric field in encoding.
        """
        field_map: dict[str, FieldMetadata] = {f.id: f for f in fields}

        # Prefer explicit measure axes
        for axis_role in ("y", "value", "size"):
            field_id = encoding.get(axis_role)
            if field_id and field_id in field_map:
                return field_map[field_id]

        # Fall back to any numeric field in encoding
        for _axis, field_id in encoding.items():
            field = field_map.get(field_id)
            if field and field.data_type in ("integer", "float", "number"):
                return field

        return None

    def _generate_title(
        self, chart_type: str, primary_measure: FieldMetadata | None
    ) -> str:
        """Generate chart title from chart_type and primary measure field label.

        Format: "{ChartType} Chart of {MeasureLabel}"
        Truncated to 200 characters. Req 9.5.
        """
        chart_type_display = chart_type.capitalize()

        if primary_measure is not None:
            title = f"{chart_type_display} Chart of {primary_measure.label}"
        else:
            title = f"{chart_type_display} Chart"

        return title[:_MAX_TITLE_LENGTH]

    def _format_error(self, message: str) -> str:
        """Format error message: plain-language, 1-500 chars, no stack traces.

        Req 18.1: Error must be 1-500 characters.
        """
        # Strip any potential stack trace indicators
        cleaned = message.strip()
        if not cleaned:
            cleaned = "An error occurred during visualization processing."
        return cleaned[:_MAX_ERROR_LENGTH]
