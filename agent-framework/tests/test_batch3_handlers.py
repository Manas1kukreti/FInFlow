import os
import sys
import pandas as pd
import numpy as np
import pytest


from finflow_agent.operations.schemas import (
    CleaningOperationPlan, TrimWhitespaceOperation, NormalizeCurrencyOperation, NormalizeNumberOperation,
    FilterOperationPlan, FilterCondition, CalculationOperationPlan, CalculationOperation,
    ReportingOperationPlan
)
from finflow_agent.operations.executor import (
    execute_cleaning_plan, execute_filter_plan, execute_calculation_plan, execute_reporting_plan
)
from finflow_agent.operations.errors import UnsupportedOperationError, OperationExecutionError, UnsafeOutputPathError, OperationValidationError

def test_unsupported_handler_fails():
    # Because of Pydantic validation, we must mock an operation to bypass it for the test
    class FakeOp:
        type = "fake_op"
        
    plan = CleaningOperationPlan(operations=[])
    plan.operations.append(FakeOp()) # type: ignore
    
    with pytest.raises(UnsupportedOperationError):
        execute_cleaning_plan(pd.DataFrame(), plan)

def test_trim_whitespace_preserves_null():
    df = pd.DataFrame({"Name": ["  John  ", None, "Alice"]})
    plan = CleaningOperationPlan(operations=[TrimWhitespaceOperation(columns=["Name"])])
    out = execute_cleaning_plan(df, plan)
    
    assert out.data["Name"].iloc[0] == "John"
    assert pd.isna(out.data["Name"].iloc[1])
    assert out.data["Name"].iloc[2] == "Alice"
    assert out.summary

def test_normalize_currency_handles_symbols():
    df = pd.DataFrame({"Price": ["$1,234.56", "€500", "-£10.5", None]})
    plan = CleaningOperationPlan(operations=[NormalizeCurrencyOperation(column="Price")])
    out = execute_cleaning_plan(df, plan)
    
    assert out.data["Price"].iloc[0] == 1234.56
    assert out.data["Price"].iloc[1] == 500.0
    assert out.data["Price"].iloc[2] == -10.5
    assert pd.isna(out.data["Price"].iloc[3])

def test_normalize_number_handles_commas():
    df = pd.DataFrame({"Qty": ["1,000", "2,500.5", None]})
    plan = CleaningOperationPlan(operations=[NormalizeNumberOperation(column="Qty")])
    out = execute_cleaning_plan(df, plan)
    
    assert out.data["Qty"].iloc[0] == 1000.0
    assert out.data["Qty"].iloc[1] == 2500.5
    assert pd.isna(out.data["Qty"].iloc[2])

def test_contains_uses_literal_matching():
    df = pd.DataFrame({"Code": ["A[B]", "AB", "C"]})
    plan = FilterOperationPlan(conditions=[FilterCondition(column="Code", operator="contains", value="[B]")])
    out = execute_filter_plan(df, plan)
    
    # regex=False means it literally matches "[B]", not the character class B
    assert len(out.data) == 1
    assert out.data["Code"].iloc[0] == "A[B]"

def test_filter_in_uses_normalized_text_matching():
    df = pd.DataFrame({"payment_method": ["PayPal", "cash", "pay pal", "credit card"]})
    plan = FilterOperationPlan(
        conditions=[
            FilterCondition(
                column="payment_method",
                operator="in",
                value=["paypal", "cash"],
            )
        ]
    )
    out = execute_filter_plan(df, plan)

    assert list(out.data["payment_method"]) == ["PayPal", "cash", "pay pal"]

def test_filter_missing_column_fails():
    df = pd.DataFrame({"A": [1]})
    plan = FilterOperationPlan(conditions=[FilterCondition(column="B", operator="gt", value=0)])
    with pytest.raises(OperationValidationError):
        execute_filter_plan(df, plan)

def test_group_sum_respects_output_column():
    df = pd.DataFrame({"Cat": ["A", "A", "B"], "Val": [10, 20, 30]})
    plan = CalculationOperationPlan(operations=[
        CalculationOperation(type="group_sum", column="Val", group_by=["Cat"], output_column="TotalVal")
    ])
    out = execute_calculation_plan(df, plan)
    
    assert "TotalVal" in out.data.columns
    assert "Val" not in out.data.columns
    assert out.data[out.data["Cat"] == "A"]["TotalVal"].iloc[0] == 30

def test_reporting_rejects_unsafe_prefix():
    df = pd.DataFrame({"A": [1]})
    plan = ReportingOperationPlan(output_format="csv")
    with pytest.raises(UnsafeOutputPathError):
        execute_reporting_plan(df, plan, os.path.abspath("."), "../evil_prefix")

def test_reporting_writes_inside_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        df = pd.DataFrame({"A": [1]})
        plan = ReportingOperationPlan(output_format="csv")
        out = execute_reporting_plan(df, plan, tmp_path, "report")
        
        expected_path = os.path.join(tmp_path, "report.csv")
        assert out.artifacts["output_file_path"] == expected_path
        assert os.path.exists(expected_path)
        assert out.summary

if __name__ == "__main__":
    test_unsupported_handler_fails()
    test_trim_whitespace_preserves_null()
    test_normalize_currency_handles_symbols()
    test_normalize_number_handles_commas()
    test_contains_uses_literal_matching()
    test_filter_missing_column_fails()
    test_group_sum_respects_output_column()
    test_reporting_rejects_unsafe_prefix()
    test_reporting_writes_inside_dir()
    print("Batch 3 handler tests passed successfully.")
