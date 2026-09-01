"""Unit tests for OperationResultReader adapter.

Tests cover field extraction, row retrieval, data shape classification,
and validation behavior per Requirements 14.1, 14.2, 14.3, 14.4.
"""

from __future__ import annotations

import os
import sys


import pytest

from finflow_agent.execution.visualization.operation_result_reader import (
    OperationResultReader,
    OperationResultReaderError,
)
from finflow_agent.execution.visualization.spec import DataShape, FieldMetadata


# --- Fixtures ---


def _make_operation_result(
    fields: list[dict], rows: list[dict] | None = None
) -> dict:
    """Helper to build an OperationResult dict."""
    result: dict = {"fields": fields}
    if rows is not None:
        result["rows"] = rows
    return result


VALID_FIELD_CATEGORY = {
    "id": "region",
    "label": "Region",
    "data_type": "string",
    "role": "category",
}

VALID_FIELD_MEASURE = {
    "id": "revenue",
    "label": "Revenue",
    "data_type": "float",
    "role": "measure",
    "unit": "USD",
    "aggregation": "sum",
}

VALID_FIELD_TIME = {
    "id": "month",
    "label": "Month",
    "data_type": "datetime",
    "role": "time",
}

VALID_FIELD_DIMENSION = {
    "id": "bin_start",
    "label": "Bin Start",
    "data_type": "float",
    "role": "dimension",
}


# --- Validation Tests (Requirement 14.2, 14.3) ---


class TestValidation:
    """Tests for _validate() ensuring proper error raising."""

    def test_raises_error_when_result_is_not_dict(self):
        with pytest.raises(OperationResultReaderError, match="must be a dictionary"):
            OperationResultReader("not a dict")  # type: ignore[arg-type]

    def test_raises_error_when_fields_key_missing(self):
        with pytest.raises(OperationResultReaderError, match="non-empty 'fields' list"):
            OperationResultReader({})

    def test_raises_error_when_fields_list_is_empty(self):
        with pytest.raises(OperationResultReaderError, match="non-empty 'fields' list"):
            OperationResultReader({"fields": []})

    def test_raises_error_when_fields_is_not_a_list(self):
        with pytest.raises(OperationResultReaderError, match="non-empty 'fields' list"):
            OperationResultReader({"fields": "invalid"})

    def test_raises_error_when_all_fields_missing_id(self):
        fields = [{"label": "X", "data_type": "float", "role": "measure"}]
        with pytest.raises(OperationResultReaderError, match="non-null id, data_type, and role"):
            OperationResultReader(_make_operation_result(fields))

    def test_raises_error_when_all_fields_missing_data_type(self):
        fields = [{"id": "x", "label": "X", "role": "measure"}]
        with pytest.raises(OperationResultReaderError, match="non-null id, data_type, and role"):
            OperationResultReader(_make_operation_result(fields))

    def test_raises_error_when_all_fields_missing_role(self):
        fields = [{"id": "x", "label": "X", "data_type": "float"}]
        with pytest.raises(OperationResultReaderError, match="non-null id, data_type, and role"):
            OperationResultReader(_make_operation_result(fields))

    def test_raises_error_when_field_values_are_none(self):
        fields = [{"id": None, "data_type": None, "role": None}]
        with pytest.raises(OperationResultReaderError):
            OperationResultReader(_make_operation_result(fields))

    def test_succeeds_with_at_least_one_valid_field(self):
        fields = [
            {"id": None, "data_type": "float", "role": "measure"},  # invalid
            VALID_FIELD_MEASURE,  # valid
        ]
        reader = OperationResultReader(_make_operation_result(fields))
        assert reader is not None


# --- get_fields() Tests (Requirement 14.1, 14.2) ---


class TestGetFields:
    """Tests for get_fields() returning FieldMetadata instances."""

    def test_returns_field_metadata_list(self):
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_MEASURE, VALID_FIELD_CATEGORY])
        )
        fields = reader.get_fields()
        assert len(fields) == 2
        assert all(isinstance(f, FieldMetadata) for f in fields)

    def test_field_metadata_attributes(self):
        reader = OperationResultReader(_make_operation_result([VALID_FIELD_MEASURE]))
        field = reader.get_fields()[0]
        assert field.id == "revenue"
        assert field.label == "Revenue"
        assert field.data_type == "float"
        assert field.role == "measure"
        assert field.unit == "USD"
        assert field.aggregation == "sum"

    def test_skips_invalid_fields(self):
        fields = [
            {"id": None, "data_type": "float", "role": "measure"},
            VALID_FIELD_CATEGORY,
        ]
        reader = OperationResultReader(_make_operation_result(fields))
        result = reader.get_fields()
        assert len(result) == 1
        assert result[0].id == "region"

    def test_uses_id_as_label_when_label_missing(self):
        field = {"id": "x_col", "data_type": "integer", "role": "measure"}
        reader = OperationResultReader(_make_operation_result([field]))
        assert reader.get_fields()[0].label == "x_col"

    def test_unit_and_aggregation_default_to_none(self):
        field = {"id": "x", "label": "X", "data_type": "integer", "role": "measure"}
        reader = OperationResultReader(_make_operation_result([field]))
        f = reader.get_fields()[0]
        assert f.unit is None
        assert f.aggregation is None


# --- get_rows() Tests (Requirement 14.4) ---


class TestGetRows:
    """Tests for get_rows() returning row data."""

    def test_returns_rows_when_present(self):
        rows = [{"revenue": 100.0, "region": "East"}]
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_MEASURE], rows=rows)
        )
        assert reader.get_rows() == rows

    def test_returns_empty_list_when_no_rows_key(self):
        reader = OperationResultReader(_make_operation_result([VALID_FIELD_MEASURE]))
        assert reader.get_rows() == []

    def test_returns_empty_list_when_rows_is_not_list(self):
        result = {"fields": [VALID_FIELD_MEASURE], "rows": "invalid"}
        reader = OperationResultReader(result)
        assert reader.get_rows() == []

    def test_returns_empty_list_preserves_valid_data_shape(self):
        """Requirement 14.4: valid fields + zero rows → empty list + valid shape."""
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_TIME, VALID_FIELD_MEASURE], rows=[])
        )
        assert reader.get_rows() == []
        assert reader.get_data_shape() == DataShape.TIME_SERIES


# --- get_data_shape() Tests (Requirement 14.1) ---


class TestGetDataShape:
    """Tests for data shape classification."""

    def test_time_series_when_time_role_present(self):
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_TIME, VALID_FIELD_MEASURE])
        )
        assert reader.get_data_shape() == DataShape.TIME_SERIES

    def test_time_series_takes_priority_over_category(self):
        """Time role takes priority even if category+measure also present."""
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_TIME, VALID_FIELD_CATEGORY, VALID_FIELD_MEASURE])
        )
        assert reader.get_data_shape() == DataShape.TIME_SERIES

    def test_categorical_series_with_category_and_measure(self):
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_CATEGORY, VALID_FIELD_MEASURE])
        )
        assert reader.get_data_shape() == DataShape.CATEGORICAL_SERIES

    def test_histogram_bins_with_dimension_and_measure(self):
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_DIMENSION, VALID_FIELD_MEASURE])
        )
        assert reader.get_data_shape() == DataShape.HISTOGRAM_BINS

    def test_scatter_points_with_two_numeric_measures_no_category(self):
        field_a = {"id": "x", "label": "X", "data_type": "float", "role": "measure"}
        field_b = {"id": "y", "label": "Y", "data_type": "integer", "role": "measure"}
        reader = OperationResultReader(_make_operation_result([field_a, field_b]))
        assert reader.get_data_shape() == DataShape.SCATTER_POINTS

    def test_not_scatter_when_category_present(self):
        """Two numeric measures + category → categorical_series, not scatter."""
        field_a = {"id": "x", "label": "X", "data_type": "float", "role": "measure"}
        field_b = {"id": "y", "label": "Y", "data_type": "integer", "role": "measure"}
        reader = OperationResultReader(
            _make_operation_result([VALID_FIELD_CATEGORY, field_a, field_b])
        )
        assert reader.get_data_shape() == DataShape.CATEGORICAL_SERIES

    def test_not_scatter_when_only_one_numeric_measure(self):
        field_a = {"id": "x", "label": "X", "data_type": "float", "role": "measure"}
        reader = OperationResultReader(_make_operation_result([field_a]))
        assert reader.get_data_shape() == DataShape.SCALAR

    def test_scalar_for_single_string_field(self):
        field = {"id": "name", "label": "Name", "data_type": "string", "role": "dimension"}
        reader = OperationResultReader(_make_operation_result([field]))
        assert reader.get_data_shape() == DataShape.SCALAR

    def test_scalar_as_default(self):
        """Single non-numeric measure without category → scalar."""
        field = {"id": "note", "label": "Note", "data_type": "string", "role": "measure"}
        reader = OperationResultReader(_make_operation_result([field]))
        assert reader.get_data_shape() == DataShape.SCALAR
