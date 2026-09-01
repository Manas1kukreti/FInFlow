"""Visualization agent for chart generation.

Produces a VisualizationSpec by consuming the OperationResult from upstream
calculation/filter steps. Delegates actual chart logic to the
VisualizationExecutor, which performs zero calculations — only structural
mapping from the operation result to a renderable chart spec.

The agent is gated by the ``ENABLE_VISUALIZATION`` config flag. When enabled,
it reads the upstream DataFrame/OperationResult, builds encoding from the
plan, and produces a VisualizationSpec that gets persisted and rendered.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel

from finflow_agent.registry import registry, AgentSpec
from finflow_agent.state import AgentResult
from finflow_agent.operations.schemas import VisualizationOperationPlan


logger = logging.getLogger(__name__)

# Canonical disabled-message used by the compiler, validator, and orchestrator
# when visualization is requested while the agent is disabled.
VISUALIZATION_DISABLED_MESSAGE = (
    "visualization_agent is not enabled in this version"
)


class VisualizationAgentParams(BaseModel):
    """Pydantic param model for the visualization agent."""

    plan: VisualizationOperationPlan


@registry.register
class VisualizationAgent:
    """Visualization agent that produces chart specs from operation results.

    When enabled, delegates to the VisualizationExecutor to generate a
    VisualizationSpec. The spec is returned as part of the AgentResult
    artifacts for downstream persistence and rendering.
    """

    spec = AgentSpec(
        name="visualization_agent",
        description=(
            "Produces a VisualizationSpec from upstream operation results. "
            "Performs zero business calculations — only structural mapping."
        ),
        stage="visualize",
        accepts=["dataframe"],
        produces=["chart_spec"],
        params_schema={
            "plan": {"type": "object"}
        },
    )
    params_model = VisualizationAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        """Execute the visualization agent.

        Reads the upstream operation result (or DataFrame), runs it through
        the VisualizationExecutor, and returns the resulting spec.
        The input DataFrame is passed through unchanged (zero-calculation).
        """
        # Pass through the input DataFrame so downstream steps (reporting) get it
        import pandas as pd
        input_df = input_data.get("input_dataframe")

        try:
            from finflow_agent.execution.visualization.executor import VisualizationExecutor
            from finflow_agent.execution.visualization.spec import VisualizationSpec

            # Extract plan params
            plan = params.get("plan", {})
            charts = plan.get("charts", []) if isinstance(plan, dict) else []

            if not charts:
                return AgentResult(
                    status="success",
                    data=input_df,
                    summary="No charts requested in visualization plan.",
                    artifacts={"visualizations": []},
                )

            executor = VisualizationExecutor()
            visualization_specs: list[dict[str, Any]] = []

            # Inject descriptions from params into chart configs
            chart_descriptions = params.get("chart_descriptions", [])

            for idx, chart_config in enumerate(charts):
                # Inject description if available
                if idx < len(chart_descriptions) and chart_descriptions[idx]:
                    chart_config = dict(chart_config)  # Don't mutate original
                    chart_config["description"] = chart_descriptions[idx]

                chart_type = chart_config.get("type", "auto")
                if not chart_type:
                    chart_type = "auto"

                # Per-chart aggregation: compute group_count/sum for this chart's
                # specific group_by column from the raw input dataframe
                chart_operation_result = self._aggregate_for_chart(input_df, chart_config)

                # Build encoding hints from chart config
                encoding_hints = self._build_encoding_hints(
                    chart_config, chart_type, chart_operation_result
                )

                source_result_id = str(uuid.uuid4())
                operation_id = f"viz_{uuid.uuid4().hex[:8]}"

                spec = executor.execute(
                    operation_result=chart_operation_result,
                    chart_type=chart_type,
                    encoding_hints=encoding_hints,
                    source_result_id=source_result_id,
                    operation_id=operation_id,
                )
                chart_options = {
                    "bar_mode": chart_config.get("bar_mode"),
                    "show_legend": chart_config.get("show_legend"),
                    "show_data_labels": chart_config.get("show_data_labels"),
                    "x_axis_title": chart_config.get("x_axis_title"),
                    "y_axis_title": chart_config.get("y_axis_title"),
                }
                spec.title = chart_config.get("title") or spec.title
                spec.options.update({k: v for k, v in chart_options.items() if v is not None})
                visualization_specs.append(spec.model_dump())

            # Determine overall status — always "success" to avoid blocking
            # the reporting agent downstream (failure isolation per Req 6.1).
            # Visualization failures are recorded in the spec artifacts.
            all_failed = all(s.get("status") == "failed" for s in visualization_specs)
            any_failed = any(s.get("status") in ("failed", "unsupported") for s in visualization_specs)

            if all_failed:
                summary = "All visualization attempts produced unsupported/failed specs."
            elif any_failed:
                summary = f"Generated {len(visualization_specs)} visualization(s), some with warnings."
            else:
                summary = f"Successfully generated {len(visualization_specs)} visualization(s)."

            return AgentResult(
                status="success",
                data=input_df,
                summary=summary,
                artifacts={"visualizations": visualization_specs},
            )

        except Exception as exc:
            logger.exception("Visualization agent failed: %s", exc)
            # Still return success with the DataFrame to not block reporting
            return AgentResult(
                status="success",
                data=input_df,
                summary=f"Visualization failed: {str(exc)[:200]}",
                artifacts={"visualizations": []},
                warnings=[f"Visualization error: {str(exc)[:300]}"],
            )

    @staticmethod
    def _build_operation_result(input_data: dict, charts: list | None = None) -> dict[str, Any]:
        """Build an OperationResult-compatible dict from agent input_data.

        The input_data typically contains a DataFrame under the key
        'input_dataframe' (set by the execution engine). We convert it into
        the OperationResult format that the VisualizationExecutor expects.

        When charts config is provided, output fields from aggregation
        operations will have their aggregation metadata set correctly.
        """
        import pandas as pd

        # Build a set of output fields that have aggregation from chart configs
        aggregated_fields: dict[str, str] = {}  # output_field -> aggregation
        if charts:
            for chart in charts:
                if isinstance(chart, dict):
                    agg = chart.get("aggregation")
                    output_field = chart.get("output_field") or "record_count"
                    if agg:
                        aggregated_fields[output_field] = agg

        df = input_data.get("input_dataframe")
        if df is None:
            df = input_data.get("df")
        if df is None:
            df = input_data.get("dataframe")
        if df is None:
            # Try to find any DataFrame in input_data values
            for v in input_data.values():
                if isinstance(v, pd.DataFrame):
                    df = v
                    break

        if df is None:
            return {"fields": [], "rows": []}

        if isinstance(df, pd.DataFrame):
            # Build field metadata from DataFrame columns
            fields = []
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    data_type = "integer"
                elif pd.api.types.is_float_dtype(dtype):
                    data_type = "float"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    data_type = "datetime"
                else:
                    data_type = "string"

                # Infer role from data type and column name
                col_lower = str(col).lower().strip()

                # Identifier detection: columns that are IDs, keys, or numbers
                # should NOT be treated as measures even if they are numeric
                is_identifier = (
                    col_lower == "id"
                    or col_lower.endswith("_id")
                    or col_lower.endswith("_number")
                    or col_lower.endswith("_key")
                    or col_lower.endswith("_no")
                    or col_lower.startswith("id_")
                    or col_lower in ("index", "row_number", "record_id")
                )

                if is_identifier:
                    role = "identifier"
                elif data_type in ("integer", "float"):
                    # Check if the field has aggregation metadata (from
                    # pre-aggregated data coming through calculation_agent)
                    role = "measure"
                elif data_type == "datetime":
                    role = "time"
                else:
                    # String columns with low cardinality → category
                    nunique = df[col].nunique()
                    total = len(df)
                    if total > 0 and nunique / total < 0.5:
                        role = "category"
                    else:
                        role = "dimension"

                fields.append({
                    "id": str(col),
                    "label": str(col),
                    "data_type": data_type,
                    "role": role,
                    "unit": None,
                    "aggregation": aggregated_fields.get(str(col)),
                })

            # Convert rows — ensure native Python types for Pydantic strict mode
            rows = []
            for record in df.head(1000).to_dict(orient="records"):
                clean_row = {}
                for k, v in record.items():
                    if hasattr(v, "item"):
                        # Convert numpy scalars to native Python types
                        clean_row[str(k)] = v.item()
                    elif pd.isna(v):
                        clean_row[str(k)] = None
                    else:
                        clean_row[str(k)] = v
                rows.append(clean_row)

            return {"fields": fields, "rows": rows}

        return {"fields": [], "rows": []}

    @staticmethod
    def _build_encoding_hints(
        chart_config: dict,
        chart_type: str,
        operation_result: dict[str, Any],
    ) -> dict[str, str] | None:
        """Build encoding hints from chart config or auto-detect from fields.

        If the chart config specifies explicit x/y fields (not "auto"),
        use those. Otherwise, auto-detect from the operation result's field
        metadata based on the chart type.

        Fields with role "identifier" are excluded from measure candidates
        to prevent IDs/keys from being selected as chart value axes.
        """
        x_field = chart_config.get("x")
        y_field = chart_config.get("y")
        series_field = chart_config.get("series")

        # If explicit fields provided, use them
        if x_field and x_field != "auto" and y_field and y_field != "auto":
            encoding = {"x": x_field, "y": y_field}
            if series_field and series_field != "auto":
                encoding["series"] = series_field
            return encoding

        # Auto-detect encoding from field metadata
        fields = operation_result.get("fields", [])
        if not fields:
            return None

        category_fields = [f for f in fields if f.get("role") in ("category", "dimension")]
        # Exclude identifiers from measure candidates
        measure_fields = [
            f for f in fields
            if f.get("role") == "measure" and f.get("role") != "identifier"
        ]
        time_fields = [f for f in fields if f.get("role") == "time"]

        if chart_type == "pie":
            # Pie needs: category for name, measure for value
            if category_fields and measure_fields:
                return {
                    "category": category_fields[0]["id"],
                    "value": measure_fields[0]["id"],
                }
        elif chart_type == "scatter":
            # Scatter needs: two measure fields
            if len(measure_fields) >= 2:
                return {
                    "x": measure_fields[0]["id"],
                    "y": measure_fields[1]["id"],
                }
        elif chart_type == "line":
            # Line: time/dimension x-axis, measure y-axis
            x = time_fields[0] if time_fields else (category_fields[0] if category_fields else None)
            y = measure_fields[0] if measure_fields else None
            if x and y:
                return {
                    "x": x["id"],
                    "y": y["id"],
                    **({"series": series_field} if series_field else {}),
                }
        else:
            # Bar, histogram, or auto: category x-axis, measure y-axis
            x = category_fields[0] if category_fields else (time_fields[0] if time_fields else None)
            y = measure_fields[0] if measure_fields else None
            if x and y:
                return {
                    "x": x["id"],
                    "y": y["id"],
                }

        # Fallback: first string-like field as x, first numeric (non-identifier) as y
        string_fields = [f for f in fields if f.get("data_type") == "string"]
        numeric_fields = [
            f for f in fields
            if f.get("data_type") in ("integer", "float") and f.get("role") != "identifier"
        ]
        if string_fields and numeric_fields:
            return {
                "x": string_fields[0]["id"],
                "y": numeric_fields[0]["id"],
                **({"series": series_field} if series_field else {}),
            }

        return None

    @staticmethod
    def _resolve_column_name(df, raw_name: str | None) -> str | None:
        if not raw_name:
            return None
        if raw_name in df.columns:
            return raw_name
        normalized = str(raw_name).strip().lower().replace(" ", "_")
        for column in df.columns:
            candidate = str(column).strip().lower().replace(" ", "_")
            if candidate == normalized:
                return str(column)
        for column in df.columns:
            candidate = str(column).strip().lower().replace(" ", "_")
            if normalized in candidate or candidate in normalized:
                return str(column)
        return None

    @staticmethod
    def _aggregate_for_chart(df, chart_config: dict) -> dict[str, Any]:
        """Independently aggregate the raw dataframe for a specific chart.

        Each chart gets its own group_by aggregation so multiple charts
        can use different grouping columns from the same source data.
        """
        import pandas as pd

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"fields": [], "rows": []}

        group_by = chart_config.get("group_by")
        measure = VisualizationAgent._resolve_column_name(df, chart_config.get("measure"))
        aggregation = chart_config.get("aggregation", "count")
        x_field = VisualizationAgent._resolve_column_name(df, chart_config.get("x", ""))
        series_field = VisualizationAgent._resolve_column_name(df, chart_config.get("series"))
        output_field = chart_config.get("output_field", "record_count")
        description = chart_config.get("description", "")
        chart_type = str(chart_config.get("type") or "auto").strip().lower()

        # Determine the grouping column
        group_cols: list[str] = []
        if group_by and isinstance(group_by, list) and group_by:
            group_cols = [
                resolved
                for col in group_by
                if (resolved := VisualizationAgent._resolve_column_name(df, str(col)))
            ]
        elif x_field and x_field in df.columns:
            group_cols = [x_field]
        if series_field and series_field in df.columns and series_field not in group_cols:
            group_cols.append(series_field)

        # If no explicit group_col, infer from chart description by matching
        # description keywords against actual column names
        if not group_cols and description:
            desc_lower = description.lower().replace("_", " ")
            col_lower_map = {c.lower().replace("_", " "): c for c in df.columns}
            # Try to find a column mentioned in the description
            for col_key, col_real in col_lower_map.items():
                if col_key in desc_lower or desc_lower in col_key:
                    # Verify it's a categorical column
                    if pd.api.types.is_string_dtype(df[col_real]) or pd.api.types.is_object_dtype(df[col_real]):
                        group_cols = [col_real]
                        break
            # Also try partial word matching
            if not group_cols:
                desc_words = [w for w in desc_lower.split() if len(w) > 3]
                for col_key, col_real in col_lower_map.items():
                    if any(word in col_key for word in desc_words):
                        if pd.api.types.is_string_dtype(df[col_real]) or pd.api.types.is_object_dtype(df[col_real]):
                            group_cols = [col_real]
                            break

        # Resolve column case-insensitively from x_field
        if not group_cols and x_field:
            resolved = VisualizationAgent._resolve_column_name(df, x_field)
            if resolved:
                group_cols = [resolved]

        if not group_cols:
            # If the plan names a measure but omits a grouping field, prefer
            # grouping by that requested column over silently defaulting to an
            # unrelated categorical column such as "gender".
            if measure and measure in df.columns:
                group_cols = [measure]

        if not group_cols:
            # Fallback: use first categorical column
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    if df[col].nunique() < len(df) * 0.5:
                        group_cols = [col]
                        break
            if not group_cols:
                group_cols = [df.columns[0]]

        if len(group_cols) > 2:
            raise ValueError("TOO_MANY_VISUALIZATION_DIMENSIONS: grouped bar charts support at most two grouping columns.")

        # Numeric measure requests with no explicit grouping should not collapse
        # onto an arbitrary categorical field. Treat them as value-frequency
        # charts instead.
        if (
            len(group_cols) == 1
            and measure
            and group_cols[0] == measure
            and measure in df.columns
            and pd.api.types.is_numeric_dtype(df[measure])
            and aggregation == "count"
        ):
            series = df[measure].dropna()
            if chart_type == "histogram" and series.nunique() > 20:
                bins = min(20, max(5, int(series.nunique() ** 0.5)))
                grouped = (
                    series.groupby(pd.cut(series, bins=bins, include_lowest=True))
                    .size()
                    .reset_index(name=output_field)
                )
                grouped.iloc[:, 0] = grouped.iloc[:, 0].astype(str)
            else:
                grouped = series.value_counts(dropna=True).sort_index().reset_index()
                grouped.columns = [measure, output_field]
            fields = [
                {
                    "id": str(measure),
                    "label": str(measure).replace("_", " ").title(),
                    "data_type": "float" if pd.api.types.is_float_dtype(df[measure]) else "integer",
                    "role": "category",
                    "aggregation": None,
                },
                {
                    "id": str(output_field),
                    "label": str(output_field).replace("_", " ").title(),
                    "data_type": "integer",
                    "role": "measure",
                    "aggregation": "count",
                },
            ]
            rows = []
            for _, row in grouped.iterrows():
                clean_row = {}
                for col in grouped.columns:
                    value = row[col]
                    if hasattr(value, "item"):
                        clean_row[str(col)] = value.item()
                    elif pd.isna(value):
                        clean_row[str(col)] = None
                    else:
                        clean_row[str(col)] = value
                rows.append(clean_row)
            return {"fields": fields, "rows": rows}

        # Perform aggregation
        if aggregation == "sum" and measure and measure in df.columns:
            grouped = df.groupby(group_cols, as_index=False)[measure].sum()
            grouped.rename(columns={measure: output_field}, inplace=True)
        elif aggregation == "mean" and measure and measure in df.columns:
            grouped = df.groupby(group_cols, as_index=False)[measure].mean()
            grouped.rename(columns={measure: output_field}, inplace=True)
        else:
            # Default: count
            grouped = df.groupby(group_cols, as_index=False).size()
            grouped.rename(columns={"size": output_field}, inplace=True)

        # Build operation_result format
        fields = []
        for idx, group_col in enumerate(group_cols):
            fields.append(
                {
                    "id": str(group_col),
                    "label": str(group_col).replace("_", " ").title(),
                    "data_type": "string",
                    "role": "category" if idx == 0 else "dimension",
                    "aggregation": None,
                }
            )
        fields.append(
            {
                "id": str(output_field),
                "label": str(output_field).replace("_", " ").title(),
                "data_type": "float" if aggregation in {"sum", "mean"} else "integer",
                "role": "measure",
                "aggregation": aggregation,
            }
        )

        rows = []
        for _, row in grouped.iterrows():
            clean_row = {}
            for col in grouped.columns:
                v = row[col]
                if hasattr(v, "item"):
                    clean_row[str(col)] = v.item()
                elif pd.isna(v):
                    clean_row[str(col)] = None
                else:
                    clean_row[str(col)] = v
            rows.append(clean_row)

        return {"fields": fields, "rows": rows}
