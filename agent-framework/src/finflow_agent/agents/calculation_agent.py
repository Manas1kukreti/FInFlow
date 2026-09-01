import logging
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from finflow_agent.registry import registry, AgentSpec
from finflow_agent.state import AgentResult
from finflow_agent.operations.schemas import CalculationOperationPlan, CalculationOperation
from finflow_agent.operations.executor import execute_calculation_plan


logger = logging.getLogger(__name__)


def _summarize_calc_ops(operations: List[Dict[str, Any]] | List[CalculationOperation]) -> str:
    parts: List[str] = []
    for op in list(operations or [])[:3]:
        if isinstance(op, dict):
            op_type = op.get("type")
            column = op.get("column")
            secondary = op.get("secondary_column")
        else:
            op_type = getattr(op, "type", None)
            column = getattr(op, "column", None)
            secondary = getattr(op, "secondary_column", None)
        label = str(op_type or "unknown")
        if column:
            label += f"(column={column})"
        if secondary:
            label += f"(secondary_column={secondary})"
        parts.append(label)
    return ", ".join(parts) if parts else "no operations provided"

class CalculationAgentParams(BaseModel):
    instruction: Optional[str] = None
    operations: List[CalculationOperation] = Field(default_factory=list)

@registry.register
class CalculationAgent:
    spec = AgentSpec(
        name="calculation_agent",
        description="Performs mathematical calculations on a dataframe.",
        stage="analyze",
        accepts=["dataframe"],
        produces=["dataframe"],
        params_schema={
            "instruction": {"type": "string"},
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "sum", "mean", "median", "min", "max", "count",
                                "count_distinct", "variance", "standard_deviation",
                                "group_sum", "group_mean", "group_count",
                                "running_total", "percentage_change", "difference", "ratio",
                                "absolute_value"
                            ]
                        },
                        "column": {"type": "string"},
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "output_column": {"type": "string"},
                        "secondary_column": {"type": "string"},
                        "sort_by": {"type": "string"},
                        "partition_by": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    )

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        df = input_data.get("input_dataframe")
        if df is None:
            return AgentResult(status="failed", error_message="input_dataframe is required. No input dataframe provided.")

        raw_ops = params.get("operations") or []
        if not raw_ops:
            return AgentResult(
                status="failed",
                error_message=(
                    "CalculationAgent requires typed calculation operations; "
                    "canonical execution does not accept raw instructions."
                ),
            )

        try:
            ops_data = []
            for op in raw_ops:
                op_type = op.get("type")
                if op_type == "group_by_sum":
                    op_type = "group_sum"
                elif op_type == "group_by_mean":
                    op_type = "group_mean"
                elif op_type == "group_by_count":
                    op_type = "group_count"

                group_by = op.get("group_by")
                if not group_by and op.get("group_by_column"):
                    group_by = [op.get("group_by_column")]

                ops_data.append({
                    "type": op_type,
                    "column": op.get("column"),
                    "output_column": op.get("output_column"),
                    "group_by": group_by,
                    "secondary_column": op.get("secondary_column"),
                    "sort_by": op.get("sort_by"),
                    "partition_by": op.get("partition_by")
                })

            plan = CalculationOperationPlan(operations=ops_data)
        except Exception as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Failed to build calculation plan from requested operations "
                    f"[{_summarize_calc_ops(raw_ops)}]: {e}"
                ),
            )

        # Strict parameter validation of final plan operations
        try:
            CalculationAgentParams.model_validate({
                "operations": plan.operations
            })
        except ValidationError as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Failed to validate calculation plan for CalculationAgent "
                    f"(operation_count={len(plan.operations)}): {e}"
                ),
            )

        try:
            output = execute_calculation_plan(df.copy(), plan)
            return AgentResult(
                status="success",
                data=output.data,
                summary=output.summary,
                metrics=output.metrics,
                operations_applied=output.operations_applied,
                warnings=output.warnings,
                artifacts=output.artifacts
            )
        except Exception as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Calculation execution failed for operations "
                    f"[{_summarize_calc_ops(plan.operations)}] on dataframe columns "
                    f"{list(df.columns)}: {e}"
                ),
            )
