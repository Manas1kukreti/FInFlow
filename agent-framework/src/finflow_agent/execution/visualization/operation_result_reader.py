"""OperationResultReader adapter for decoupling visualization from OperationResult shape.

Provides a stable interface for the Visualization_Executor to access field metadata,
row data, and data shape classification from an OperationResult payload without
coupling to its internal structure.

Requirements: 14.1, 14.2, 14.3, 14.4
"""

from __future__ import annotations

from typing import Any

from finflow_agent.execution.visualization.spec import DataShape, FieldMetadata


class OperationResultReaderError(Exception):
    """Raised when the OperationResult fails validation.

    The Visualization_Executor handles this by setting status to "failed"
    with reason_code "invalid_source_data".
    """


class OperationResultReader:
    """Adapter that decouples the Visualization_Executor from the raw OperationResult.

    Reads field metadata and rows from an OperationResult dictionary, validates
    structural requirements, and classifies the data shape based on field roles.

    Args:
        operation_result: A dictionary containing "fields" (list of field metadata dicts)
            and optionally "rows" (list of row dictionaries).

    Raises:
        OperationResultReaderError: If the operation_result lacks valid field metadata.

    Requirements: 14.1, 14.2, 14.3, 14.4
    """

    def __init__(self, operation_result: dict[str, Any]) -> None:
        self._validate(operation_result)
        self._result = operation_result

    def get_fields(self) -> list[FieldMetadata]:
        """Return the list of FieldMetadata from the operation result.

        Each field dict is converted to a FieldMetadata dataclass instance.
        Only fields with valid (non-null) id, data_type, and role are included.

        Returns:
            List of FieldMetadata instances.

        Requirements: 14.1, 14.2
        """
        raw_fields = self._result.get("fields", [])
        result: list[FieldMetadata] = []
        for field in raw_fields:
            if not isinstance(field, dict):
                continue
            fid = field.get("id")
            data_type = field.get("data_type")
            role = field.get("role")
            if fid is None or data_type is None or role is None:
                continue
            result.append(
                FieldMetadata(
                    id=fid,
                    label=field.get("label", fid),
                    data_type=data_type,
                    role=role,
                    unit=field.get("unit"),
                    aggregation=field.get("aggregation"),
                )
            )
        return result

    def get_rows(self) -> list[dict[str, Any]]:
        """Return the data rows from the operation result.

        Returns an empty list if no rows are present, which is valid per
        Requirement 14.4.

        Returns:
            List of row dictionaries (may be empty).

        Requirements: 14.4
        """
        rows = self._result.get("rows", [])
        if not isinstance(rows, list):
            return []
        return rows

    def get_data_shape(self) -> DataShape:
        """Classify the data shape based on field roles.

        Classification rules (evaluated in priority order):
        1. If any field has role "time" → TIME_SERIES
        2. If fields include at least one "category" role AND one "measure" role
           → CATEGORICAL_SERIES
        3. If fields represent precomputed bin boundaries and frequencies
           (at least one field with role "dimension" and at least one with role
           "measure" and data_type integer or float) → HISTOGRAM_BINS
        4. If exactly two numeric measure fields exist and no category field
           → SCATTER_POINTS
        5. Otherwise → SCALAR

        Returns:
            A DataShape enum value.

        Requirements: 14.1
        """
        fields = self.get_fields()

        roles = [f.role for f in fields]
        has_time = "time" in roles
        has_category = "category" in roles
        has_measure = "measure" in roles
        has_dimension = "dimension" in roles

        # Rule 1: Any field with role "time" → time_series
        if has_time:
            return DataShape.TIME_SERIES

        # Rule 2: At least one "category" + one "measure" → categorical_series
        if has_category and has_measure:
            return DataShape.CATEGORICAL_SERIES

        # Rule 3: Precomputed bin boundaries (dimension) + frequencies (numeric measure)
        # → histogram_bins
        if has_dimension and has_measure:
            numeric_measures = [
                f for f in fields
                if f.role == "measure" and f.data_type in ("integer", "float")
            ]
            if numeric_measures:
                return DataShape.HISTOGRAM_BINS

        # Rule 4: Exactly two numeric measure fields, no category → scatter_points
        numeric_measures = [
            f for f in fields
            if f.role == "measure" and f.data_type in ("integer", "float")
        ]
        if len(numeric_measures) == 2 and not has_category:
            return DataShape.SCATTER_POINTS

        # Rule 5: Otherwise → scalar
        return DataShape.SCALAR

    def _validate(self, result: dict[str, Any]) -> None:
        """Validate that the OperationResult contains at least one valid field.

        A valid field has non-null id, data_type, and role attributes.

        Raises:
            OperationResultReaderError: If no valid fields are found.

        Requirements: 14.2, 14.3
        """
        if not isinstance(result, dict):
            raise OperationResultReaderError(
                "OperationResult must be a dictionary."
            )

        raw_fields = result.get("fields")
        if not raw_fields or not isinstance(raw_fields, list):
            raise OperationResultReaderError(
                "OperationResult must contain a non-empty 'fields' list."
            )

        # Check that at least one field has non-null id, data_type, and role
        has_valid_field = False
        for field in raw_fields:
            if not isinstance(field, dict):
                continue
            if (
                field.get("id") is not None
                and field.get("data_type") is not None
                and field.get("role") is not None
            ):
                has_valid_field = True
                break

        if not has_valid_field:
            raise OperationResultReaderError(
                "OperationResult must contain at least one field with "
                "non-null id, data_type, and role."
            )
