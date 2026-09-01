import os
import sys
import pandas as pd

# Add src to path

from finflow_agent.operations.schemas import (
    CleaningOperationPlan,
    TrimWhitespaceOperation,
    DropDuplicatesOperation,
    FillNullsOperation,
    DropNullsOperation,
    FilterOperationPlan,
    FilterCondition,
    CalculationOperationPlan,
    CalculationOperation,
    VisualizationOperationPlan,
    ChartSpec,
    ReportingOperationPlan
)
from finflow_agent.operations.executor import (
    execute_cleaning_plan, 
    execute_filter_plan,
    execute_calculation_plan,
    execute_visualization_plan,
    execute_reporting_plan
)

def test_execute_cleaning_plan():
    df = pd.DataFrame({
        "Name": [" Alice ", "Bob", "Alice ", None],
        "Score": [10, 20, 10, None]
    })
    
    plan = CleaningOperationPlan(
        operations=[
            TrimWhitespaceOperation(columns=["Name"]),
            DropDuplicatesOperation(keep="first"),
            FillNullsOperation(columns=["Name"], strategy="constant", value="Unknown"),
            FillNullsOperation(columns=["Score"], strategy="zero")
        ]
    )
    
    output = execute_cleaning_plan(df, plan)
    clean_df = output.data
    
    # Assertions
    assert len(clean_df) == 3 # Dropped 1 duplicate
    assert clean_df.iloc[0]["Name"] == "Alice" # Trimmed
    assert clean_df.iloc[2]["Name"] == "Unknown" # Filled null string
    assert clean_df.iloc[2]["Score"] == 0.0 # Filled null numeric
    
    assert len(output.operations_applied) == 4
    print("Cleaning execution test passed.")


def test_execute_drop_nulls_treats_empty_strings_as_missing():
    df = pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "education_level": ["PhD", "Master", ""],
        "marital_status": ["Single", None, "Single"],
    })

    plan = CleaningOperationPlan(
        operations=[
            DropNullsOperation(columns=None, how="any"),
        ]
    )

    output = execute_cleaning_plan(df, plan)
    clean_df = output.data

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["gender"] == "Female"
    assert clean_df.iloc[0]["education_level"] == "PhD"
    assert clean_df.iloc[0]["marital_status"] == "Single"

def test_execute_filter_plan():
    df = pd.DataFrame({
        "Status": ["Paid", "Pending", "Failed", "Paid"],
        "Amount": [100, 200, 50, 300]
    })
    
    plan = FilterOperationPlan(
        conditions=[
            FilterCondition(column="Status", operator="eq", value="Paid"),
            FilterCondition(column="Amount", operator="gt", value=150)
        ],
        logic="OR",
        select_columns=["Amount"]
    )
    
    output = execute_filter_plan(df, plan)
    filter_df = output.data
    
    # Assertions
    # Status=Paid (rows 0, 3) OR Amount > 150 (rows 1, 3) => rows 0, 1, 3 (Total 3 rows)
    assert len(filter_df) == 3
    assert list(filter_df.columns) == ["Amount"] # Projection
    assert 50 not in filter_df["Amount"].values
    
    print("Filter execution test passed.")

def test_execute_calculation_plan():
    df = pd.DataFrame({
        "Category": ["A", "A", "B"],
        "Revenue": [100, 200, 300]
    })
    
    plan = CalculationOperationPlan(
        operations=[
            CalculationOperation(type="sum", column="Revenue", output_column="total_rev"),
            CalculationOperation(type="group_sum", column="Revenue", group_by=["Category"])
        ]
    )
    
    output = execute_calculation_plan(df, plan)
    
    assert output.metrics["total_rev"] == 600
    # Data should be grouped now
    assert len(output.data) == 2
    assert output.data[output.data["Category"] == "A"]["sum_Revenue"].iloc[0] == 300
    print("Calculation execution test passed.")

def test_execute_visualization_plan():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Value": [100]})
    
    plan = VisualizationOperationPlan(
        charts=[
            ChartSpec(type="line", x="Date", y="Value", title="Trend")
        ]
    )
    
    output = execute_visualization_plan(df, plan)
    assert "chart_line_Date_Value" in output.artifacts
    assert output.artifacts["chart_line_Date_Value"]["title"] == "Trend"
    print("Visualization execution test passed.")

def test_execute_reporting_plan():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        df = pd.DataFrame({"ID": [1, 2]})
        plan = ReportingOperationPlan(output_format="csv")
        
        output = execute_reporting_plan(df, plan, output_dir=str(tmp_path), file_prefix="test_report")
        assert "output_file_path" in output.artifacts
        assert os.path.exists(output.artifacts["output_file_path"])
        print("Reporting execution test passed.")

if __name__ == "__main__":
    import pytest
    import tempfile
    
    test_execute_cleaning_plan()
    test_execute_filter_plan()
    test_execute_calculation_plan()
    test_execute_visualization_plan()
    
    with tempfile.TemporaryDirectory() as tmp_path:
        test_execute_reporting_plan(tmp_path)
        
    print("All advanced executor tests passed successfully.")
