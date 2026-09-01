import os
import sys
import pandas as pd
from pathlib import Path
import pytest


from finflow_agent.state import AgentResult, ExecutionOutput
from finflow_agent.tools.dataframe_profile import profile_dataframe
from finflow_agent.tools.path_safety import get_safe_output_path
from finflow_agent.tools.serialization import safe_model_dump, safe_model_dump_json
from finflow_agent.operations.errors import UnsafeOutputPathError

def test_agentresult_serialization():
    result = AgentResult(
        status="success",
        summary="Test summary",
        operations_applied=[{"type": "drop_nulls"}]
    )
    d = safe_model_dump(result)
    assert d["status"] == "success"
    assert d["summary"] == "Test summary"
    assert len(d["operations_applied"]) == 1

def test_executionoutput_independence():
    out1 = ExecutionOutput(data=None)
    out2 = ExecutionOutput(data=None)
    
    out1.warnings.append("Warning 1")
    out1.metrics["k1"] = "v1"
    
    assert len(out2.warnings) == 0
    assert "k1" not in out2.metrics

def test_dataframe_profile():
    df = pd.DataFrame({
        "ID": [1, 2, 2, None],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", None])
    })
    
    profile = profile_dataframe(df)
    assert profile["row_count"] == 4
    assert profile["column_count"] == 2
    assert profile["duplicate_row_count"] == 0 # no full duplicate row, wait row 2 and 3 are not duplicates.
    assert profile["null_counts"]["ID"] == 1
    assert "Date" in profile["likely_date_columns"]

def test_path_safety():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        base_dir = str(tmp_path)
        
        # Valid
        safe_p = get_safe_output_path(base_dir, "report.csv")
        assert str(safe_p.parent) == base_dir
        assert safe_p.name == "report.csv"
        
        # Traversal attempt should now explicitly raise UnsafeOutputPathError
        with pytest.raises(UnsafeOutputPathError):
            get_safe_output_path(base_dir, "../evil.xlsx")
        
        # Absolute path attempt should also fail if os.path.basename modifies it
        with pytest.raises(UnsafeOutputPathError):
            get_safe_output_path(base_dir, "/root/evil.xlsx")

if __name__ == "__main__":
    test_agentresult_serialization()
    test_executionoutput_independence()
    test_dataframe_profile()
    test_path_safety()
    print("Batch 1 foundation tests passed successfully.")
