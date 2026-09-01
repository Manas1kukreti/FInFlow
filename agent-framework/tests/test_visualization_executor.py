"""Unit tests for VisualizationExecutor.

Tests cover:
- chart_type normalization (null/empty → "auto")
- chart_type validation against supported set
- OperationResultReader integration (invalid source data → "failed")
- Field reference validation (existence, data_type, role)
- Chart compatibility validation delegation
- Auto chart type selection (data_shape → chart_type)
- Zero-calculation: rows copied verbatim
- Title generation
- Error message formatting (1-500 chars, no stack traces)
- Unhandled exception catch-all → "failed" with internal_error style message
"""

from __future__ import annotations

import os
import sys


import pytest

from finflow_agent.execution.visualization.executor import VisualizationExecutor


@pytest.fixture
def executor() -> VisualizationExecutor:
    return VisualizationExecutor()


def _make_operation_result(
    fields: list[dict], rows: list[dict] | None = None
) -> dict:
    """Helper to build a valid OperationResult dict."""
    result: dict = {"fields": fields}
    if rows is not None:
        result["rows"] = rows
    return result


# --- Fixtures for common field/row patterns ---

CATEGORY_FIELD = {"id": "cat", "label": "Category", "data_type": "string", "role": "category"}
MEASURE_FIELD = {"id": "val", "label": "Revenue", "data_type": "float", "role": "measure"}
TIME_FIELD = {"id": "ts", "label": "Timestamp", "data_type": "datetime", "role": "time"}
DIMENSION_FIELD = {"id": "bin", "label": "Bin", "data_type": "string", "role": "dimension"}
MEASURE_INT_FIELD = {"id": "freq", "label": "Frequency", "data_type": "integer", "role": "measure"}


class TestChartTypeNormalization:
    """Req 15.4: null/empty/missing chart_type → 'auto'."""

    def test_none_chart_type_becomes_auto(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, None, {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.chart_type != "auto"  # resolved from auto
        assert spec.status == "ready"

    def test_empty_string_chart_type_becomes_auto(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.chart_type != "auto"  # resolved
        assert spec.status == "ready"


class TestChartTypeValidation:
    """Req 15.1, 15.2, 15.3: Only exact lowercase supported types accepted."""

    def test_unsupported_chart_type_returns_unsupported(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "radar", {}, "src1", "op1")
        assert spec.status == "unsupported"
        assert "not supported" in spec.error.lower()

    def test_uppercase_chart_type_unsupported(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "Bar", {}, "src1", "op1")
        assert spec.status == "unsupported"

    def test_mixed_case_chart_type_unsupported(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "LINE", {}, "src1", "op1")
        assert spec.status == "unsupported"

    @pytest.mark.parametrize("chart_type", ["bar", "line", "pie", "scatter", "histogram"])
    def test_supported_chart_types_accepted(self, executor: VisualizationExecutor, chart_type: str):
        # Build a compatible data set for each type
        if chart_type == "line":
            fields = [TIME_FIELD, MEASURE_FIELD]
            rows = [{"ts": "2024-01-01", "val": 10.0}, {"ts": "2024-02-01", "val": 20.0}]
            encoding = {"x": "ts", "y": "val"}
        elif chart_type == "bar":
            fields = [CATEGORY_FIELD, MEASURE_FIELD]
            rows = [{"cat": "A", "val": 10.0}]
            encoding = {"x": "cat", "y": "val"}
        elif chart_type == "pie":
            fields = [CATEGORY_FIELD, MEASURE_FIELD]
            rows = [{"cat": "A", "val": 10.0}]
            encoding = {"x": "cat", "y": "val"}
        elif chart_type == "scatter":
            fields = [
                {"id": "x_val", "label": "X", "data_type": "float", "role": "measure"},
                {"id": "y_val", "label": "Y", "data_type": "float", "role": "measure"},
            ]
            rows = [{"x_val": 1.0, "y_val": 2.0}, {"x_val": 3.0, "y_val": 4.0}]
            encoding = {"x": "x_val", "y": "y_val"}
        else:  # histogram
            fields = [DIMENSION_FIELD, MEASURE_INT_FIELD]
            rows = [{"bin": "0-10", "freq": 5}]
            encoding = {"x": "bin", "y": "freq"}

        op = _make_operation_result(fields, rows)
        spec = executor.execute(op, chart_type, encoding, "src1", "op1")
        # Should not be unsupported due to chart_type
        assert spec.chart_type == chart_type


class TestSourceDataValidation:
    """Req 14.2, 14.3: Invalid source data → status 'failed'."""

    def test_missing_fields_returns_failed(self, executor: VisualizationExecutor):
        spec = executor.execute({}, "bar", {}, "src1", "op1")
        assert spec.status == "failed"
        assert spec.data == []
        assert spec.encoding == {}
        assert spec.error is not None

    def test_empty_fields_list_returns_failed(self, executor: VisualizationExecutor):
        spec = executor.execute({"fields": []}, "bar", {}, "src1", "op1")
        assert spec.status == "failed"

    def test_all_null_fields_returns_failed(self, executor: VisualizationExecutor):
        op = {"fields": [{"id": None, "data_type": None, "role": None}]}
        spec = executor.execute(op, "bar", {}, "src1", "op1")
        assert spec.status == "failed"


class TestFieldReferenceValidation:
    """Req 17.1-17.6: Field references validated before chart compatibility."""

    def test_nonexistent_field_returns_unsupported(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        encoding = {"x": "cat", "y": "nonexistent"}
        spec = executor.execute(op, "bar", encoding, "src1", "op1")
        assert spec.status == "unsupported"
        assert "nonexistent" in spec.error

    def test_measure_axis_with_string_type_returns_unsupported(self, executor: VisualizationExecutor):
        string_field = {"id": "name", "label": "Name", "data_type": "string", "role": "measure"}
        op = _make_operation_result([CATEGORY_FIELD, string_field], [{"cat": "A", "name": "test"}])
        encoding = {"x": "cat", "y": "name"}
        spec = executor.execute(op, "bar", encoding, "src1", "op1")
        assert spec.status == "unsupported"
        assert "name" in spec.error

    def test_empty_encoding_passes_validation(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        # Empty encoding shouldn't fail field validation
        spec = executor.execute(op, "bar", {}, "src1", "op1")
        # May fail chart compatibility, but not field reference validation
        assert spec.status in ("ready", "unsupported")


class TestAutoChartTypeSelection:
    """Req 7.1-7.4: Auto-selection based on data_shape."""

    def test_time_series_selects_line(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [TIME_FIELD, MEASURE_FIELD],
            [{"ts": "2024-01-01", "val": 10.0}, {"ts": "2024-02-01", "val": 20.0}],
        )
        spec = executor.execute(op, "auto", {"x": "ts", "y": "val"}, "src1", "op1")
        assert spec.chart_type == "line"

    def test_categorical_series_selects_bar(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "auto", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.chart_type == "bar"

    def test_histogram_bins_selects_histogram(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [DIMENSION_FIELD, MEASURE_INT_FIELD],
            [{"bin": "0-10", "freq": 5}],
        )
        spec = executor.execute(op, "auto", {"x": "bin", "y": "freq"}, "src1", "op1")
        assert spec.chart_type == "histogram"

    def test_scatter_points_selects_scatter(self, executor: VisualizationExecutor):
        fields = [
            {"id": "x_val", "label": "X", "data_type": "float", "role": "measure"},
            {"id": "y_val", "label": "Y", "data_type": "float", "role": "measure"},
        ]
        op = _make_operation_result(
            fields,
            [{"x_val": 1.0, "y_val": 2.0}, {"x_val": 3.0, "y_val": 4.0}],
        )
        spec = executor.execute(op, "auto", {"x": "x_val", "y": "y_val"}, "src1", "op1")
        assert spec.chart_type == "scatter"

    def test_scalar_defaults_to_bar(self, executor: VisualizationExecutor):
        # Single measure field with no category/time → scalar → default bar
        fields = [{"id": "total", "label": "Total", "data_type": "float", "role": "measure"}]
        op = _make_operation_result(fields, [{"total": 100.0}])
        spec = executor.execute(op, "auto", {"y": "total"}, "src1", "op1")
        # Scalar maps to "bar" default
        assert spec.chart_type == "bar"

    def test_auto_never_selects_pie(self, executor: VisualizationExecutor):
        """Req 7.2: Auto SHALL NOT select pie."""
        # All data shapes that could potentially map to pie
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "auto", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.chart_type != "pie"

    def test_auto_resolved_type_recorded_not_auto(self, executor: VisualizationExecutor):
        """Req 7.4: The resolved type (not 'auto') is in the spec."""
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "auto", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.chart_type != "auto"


class TestZeroCalculation:
    """Req 3.1, 3.2, 3.4: Rows copied verbatim, no mutation."""

    def test_data_matches_source_rows_exactly(self, executor: VisualizationExecutor):
        rows = [{"cat": "A", "val": 10.5}, {"cat": "B", "val": 20.3}]
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], rows)
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.status == "ready"
        assert spec.data == rows

    def test_empty_rows_produces_empty_data(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [])
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "src1", "op1")
        # Will fail chart compatibility (no rows), but data should be empty
        assert spec.data == []


class TestTitleGeneration:
    """Req 9.5: Title from chart_type + primary measure label, max 200 chars."""

    def test_title_format(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.title == "Bar Chart of Revenue"

    def test_title_max_200_chars(self, executor: VisualizationExecutor):
        long_label_field = {
            "id": "long", "label": "A" * 300, "data_type": "float", "role": "measure"
        }
        op = _make_operation_result(
            [CATEGORY_FIELD, long_label_field],
            [{"cat": "A", "long": 10.0}],
        )
        spec = executor.execute(op, "bar", {"x": "cat", "y": "long"}, "src1", "op1")
        assert len(spec.title) <= 200

    def test_title_non_empty_for_all_statuses(self, executor: VisualizationExecutor):
        # Unsupported status
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "INVALID", {}, "src1", "op1")
        assert spec.title != ""
        assert len(spec.title) > 0


class TestSpecContract:
    """Req 9.1-9.4: VisualizationSpec structural contract."""

    def test_ready_spec_has_data_and_encoding(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.status == "ready"
        assert len(spec.data) >= 1
        assert len(spec.encoding) >= 1
        assert spec.error is None

    def test_unsupported_spec_has_empty_data_and_encoding(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "radar", {}, "src1", "op1")
        assert spec.status == "unsupported"
        assert spec.data == []
        assert spec.encoding == {}
        assert spec.error is not None

    def test_failed_spec_has_empty_data_and_encoding(self, executor: VisualizationExecutor):
        spec = executor.execute({}, "bar", {}, "src1", "op1")
        assert spec.status == "failed"
        assert spec.data == []
        assert spec.encoding == {}
        assert spec.error is not None

    def test_schema_version_is_1_0(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "src1", "op1")
        assert spec.schema_version == "1.0"

    def test_operation_id_and_source_result_id_preserved(self, executor: VisualizationExecutor):
        op = _make_operation_result(
            [CATEGORY_FIELD, MEASURE_FIELD],
            [{"cat": "A", "val": 10.0}],
        )
        spec = executor.execute(op, "bar", {"x": "cat", "y": "val"}, "my_src", "my_op")
        assert spec.operation_id == "my_op"
        assert spec.source_result_id == "my_src"


class TestErrorMessages:
    """Req 18.1: Error messages 1-500 chars, plain language, no stack traces."""

    def test_error_message_length_within_bounds(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "unknown_type", {}, "src1", "op1")
        assert spec.error is not None
        assert 1 <= len(spec.error) <= 500

    def test_error_message_is_plain_language(self, executor: VisualizationExecutor):
        op = _make_operation_result([CATEGORY_FIELD, MEASURE_FIELD], [{"cat": "A", "val": 10.0}])
        spec = executor.execute(op, "radar", {}, "src1", "op1")
        # Should not contain stack trace markers
        assert "Traceback" not in spec.error
        assert "File \"" not in spec.error


class TestUnhandledException:
    """Catch-all for unhandled exceptions → 'failed' with internal_error style."""

    def test_unhandled_exception_returns_failed(self, executor: VisualizationExecutor):
        # Pass something that will cause an internal error during reading
        # (a non-dict that passes the type check but fails internally)
        spec = executor.execute(None, "bar", {}, "src1", "op1")  # type: ignore
        assert spec.status == "failed"
        assert spec.error is not None
        assert 1 <= len(spec.error) <= 500
