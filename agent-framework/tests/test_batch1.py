import os
import sys
import pandas as pd
import json

# Add src to path

from finflow_agent.state import AgentResult
from finflow_agent.tools.dataframe_profile import profile_dataframe

def test_agent_result_serialization():
    # Create result with new optional/extended fields
    result = AgentResult(
        status="success",
        data="dummy_data",
        summary="Operation successful",
        metrics={"duration_ms": 150},
        operations_applied=[{"type": "drop_duplicates", "rows_removed": 5}],
        warnings=["Some rows were empty"],
        artifacts={"chart_1": "base64_encoded_string"}
    )
    
    # Serialize to dict and json
    serialized = result.model_dump()
    json_str = result.model_dump_json()
    
    assert serialized["status"] == "success"
    assert serialized["summary"] == "Operation successful"
    assert serialized["metrics"]["duration_ms"] == 150
    assert len(serialized["operations_applied"]) == 1
    assert "drop_duplicates" in json_str
    print("AgentResult serialization test passed.")

def test_dataframe_profiling():
    # Create dummy dataframe
    data = {
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", None]),
        "Revenue": [100.50, 200.75, None, 400.00],
        "Status": ["Paid", "Pending", "Paid", "Failed"],
        "Notes": ["Test note 1", "Test note 2", "Test note 3", "Test note 4"]
    }
    df = pd.DataFrame(data)
    
    # Add a duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    profile = profile_dataframe(df)
    
    # Assertions
    assert profile["row_count"] == 5
    assert profile["column_count"] == 4
    assert profile["duplicate_row_count"] == 1
    assert profile["null_counts"]["Revenue"] == 1
    assert "Date" in profile["likely_date_columns"]
    assert "Revenue" in profile["likely_currency_columns"] or "Revenue" in profile["likely_numeric_columns"]
    assert "Status" in profile["likely_categorical_columns"]
    
    # Ensure it's JSON serializable
    json.dumps(profile)
    
    print("DataFrame profiling test passed.")

if __name__ == "__main__":
    test_agent_result_serialization()
    test_dataframe_profiling()
    print("All Batch 1 tests passed successfully.")
