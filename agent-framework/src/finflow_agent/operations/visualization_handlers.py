import pandas as pd
from typing import Dict, Any
from finflow_agent.operations.schemas import ChartSpec
from finflow_agent.tools.serialization import safe_model_dump

def handle_chart_spec(df: pd.DataFrame, chart: ChartSpec) -> Dict[str, Any]:
    y_str = chart.y if isinstance(chart.y, str) else "_".join(chart.y)
    chart_id = f"chart_{chart.type}_{chart.x}_{y_str}"
    
    # Simple validation that columns exist
    if chart.x not in df.columns:
        raise ValueError(f"Chart x column '{chart.x}' not found.")
        
    y_cols = [chart.y] if isinstance(chart.y, str) else chart.y
    for y in y_cols:
        if y not in df.columns:
            raise ValueError(f"Chart y column '{y}' not found.")
            
    return {"chart_id": chart_id, "spec": safe_model_dump(chart)}

VISUALIZATION_HANDLERS = {
    # Since all charts are handled generally the same right now
    # We map the types just for validation
    "bar": handle_chart_spec,
    "line": handle_chart_spec,
    "pie": handle_chart_spec,
    "scatter": handle_chart_spec,
    "area": handle_chart_spec,
    "stacked_bar": handle_chart_spec
}
