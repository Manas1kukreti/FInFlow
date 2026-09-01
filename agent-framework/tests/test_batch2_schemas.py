import os
import sys
import pandas as pd
import pytest
from pydantic import ValidationError


from finflow_agent.operations.schemas import (
    CleaningOperationPlan,
    TrimWhitespaceOperation,
    FilterOperationPlan,
    FilterCondition,
    CalculationOperationPlan,
    CalculationOperation,
    VisualizationOperationPlan,
    ChartSpec,
    ReportingOperationPlan
)
from finflow_agent.operations.validators import validate_columns_exist
from finflow_agent.operations.errors import OperationValidationError

def test_valid_cleaning_plan():
    plan = CleaningOperationPlan(
        operations=[
            TrimWhitespaceOperation(columns=["Name"])
        ]
    )
    assert len(plan.operations) == 1

def test_invalid_cleaning_operation():
    with pytest.raises(ValidationError):
        # Type "trim_whitespace" is forced by literal, sending "invalid_type" should fail
        TrimWhitespaceOperation(type="invalid_type", columns=["Name"])

def test_valid_filter_plan():
    plan = FilterOperationPlan(
        conditions=[
            FilterCondition(column="Age", operator="gt", value=18)
        ]
    )
    assert plan.conditions[0].operator == "gt"

def test_filter_between_requires_values():
    with pytest.raises(ValidationError):
        FilterCondition(column="Age", operator="between", value=18) # Missing value_to

def test_filter_in_requires_list():
    with pytest.raises(ValidationError):
        FilterCondition(column="Status", operator="in", value="Active") # Should be list

def test_reporting_rejects_unsupported_format():
    with pytest.raises(ValidationError):
        ReportingOperationPlan(output_format="docx")

def test_visualization_rejects_unsupported_chart():
    with pytest.raises(ValidationError):
        ChartSpec(type="radar", x="Date", y="Value", title="Radar Chart")

def test_calculation_rejects_unsupported_operation():
    with pytest.raises(ValidationError):
        CalculationOperation(type="integral", column="Revenue")

def test_column_validator_catches_missing():
    df = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(OperationValidationError):
        validate_columns_exist(df, ["A", "C"])
        
    # Should pass
    validate_columns_exist(df, ["A"])

if __name__ == "__main__":
    test_valid_cleaning_plan()
    test_invalid_cleaning_operation()
    test_valid_filter_plan()
    test_filter_between_requires_values()
    test_filter_in_requires_list()
    test_reporting_rejects_unsupported_format()
    test_visualization_rejects_unsupported_chart()
    test_calculation_rejects_unsupported_operation()
    test_column_validator_catches_missing()
    print("Batch 2 schema tests passed successfully.")
