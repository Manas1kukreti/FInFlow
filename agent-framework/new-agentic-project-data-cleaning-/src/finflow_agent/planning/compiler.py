"""Deterministic compiler that turns validated intent into a fixed-shape
``ExecutionPlan``.

This compiler is the single source of truth for the canonical pipeline shape
described by the agent-pipeline-hardening spec (Component 6). It enforces:

* ``ingestion_agent`` is always the first step and ``reporting_agent`` is
  always the last step.
* A ``cleaning_agent`` step running in ``mode="clean"`` is emitted only when
  the intent requested cleaning AND supplied a ``cleaning_plan``. Its output
  is published under the canonical key ``df_cleaned``.
* When filtering is requested without cleaning, a non-destructive
  ``filter_prep`` step is inserted (realized as a ``cleaning_agent``
  invocation with ``mode="filter_prep"`` and the seven safe normalization
  operations). Its output is published under ``df_filter_prepared`` and the
  subsequent ``filter_agent`` step reads from that key, never from the raw
  ``df_ingested`` output.
* A ``visualization_agent`` step is only emitted when visualization is
  enabled at the process level. Otherwise the compiler raises
  :class:`VisualizationDisabledError`.
 * Every emitted ``output_key`` is drawn from the canonical set or one of the
   deterministic visualization branch families
   ``{df_calc_viz_<n>, df_visualized_<n>}``.

Requirements satisfied: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10,
2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 9.2.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from finflow_agent.agents.visualization_agent import VISUALIZATION_DISABLED_MESSAGE
from finflow_agent.contract_registry import (
    validate_operator,
    validate_action_kind,
    resolve_semantic_operator,
    resolve_semantic_operation_type,
    InvalidOperatorError,
    InvalidActionKindError,
    UnmappedSemanticTypeError,
)
from finflow_agent.operations.schemas import (
    AbsoluteValueOperation,
    CalculationOperation,
    CalculationOperationPlan,
    ChartSpec,
    CleaningOperationPlan,
    DropDuplicatesOperation,
    DropNullsOperation,
    FillNullsOperation,
    FilterCondition,
    FilterOperationPlan,
    NormalizeColumnNamesOperation,
    NormalizeDateOperation,
    NormalizeTextCaseOperation,
    RemoveCommasFromNumbersOperation,
    RemoveEmptyRowsOperation,
    StripCurrencySymbolsOperation,
    TrimWhitespaceOperation,
    VisualizationOperationPlan,
)
from finflow_agent.planning.canonical_intent import (
    CANONICAL_INTENT_SCHEMA_VERSION,
    CanonicalIntent,
    CalculateIntent,
    CleanIntent,
    DropColumnsIntent,
    FilterRowsIntent,
    LimitRowsIntent,
    ProjectColumnsIntent,
    ReportIntent,
    SortRowsIntent,
    VisualizeIntent,
    UnresolvedColumnReference,
)
from finflow_agent.planning.intent_schema import PlanIntent
from finflow_agent.state import ExecutionPlan, PlanStep
from finflow_agent.tools.config import get_enable_visualization

# ---------------------------------------------------------------------------
# Public types and constants
# ---------------------------------------------------------------------------


class SemanticCompilationError(Exception):
    """Raised when the compiler detects a contract violation during compilation.

    This error wraps contract-registry validation failures (invalid operators
    or invalid action kinds) so downstream consumers of the compiler have a
    single exception type to catch for all compilation failures. The message
    includes the invalid value and valid options for fast debugging.
    """


class VisualizationDisabledError(Exception):
    """Raised when a ``PlanIntent`` requests visualization while the
    visualization agent is disabled at the process level.

    The orchestrator catches this exception and converts it into a quarantine
    result (Requirement 9.3 / 11.6); it MUST NOT leak past the orchestrator.
    The canonical message is built from
    :data:`VISUALIZATION_DISABLED_MESSAGE` so the validator, compiler, and
    disabled agent placeholder all surface the same wording.
    """


# Exact message the compiler raises. Composed from the shared constant so the
# wording stays in lockstep with the validator and the visualization agent
# scaffold.
VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE: str = (
    f"Visualization was requested, but {VISUALIZATION_DISABLED_MESSAGE}"
)

# Canonical ``output_key`` set that the compiler is permitted to emit. The
# visualization branch also allows deterministic suffixed variants checked by
# ``_is_canonical_output_key`` below.
CANONICAL_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "df_ingested",
        "df_cleaned",
        "df_filter_prepared",
        "df_filtered",
        "df_calculated",
        "df_calc_viz",
        "df_calc_viz_1",
        "df_calc_viz_2",
        "df_calc_viz_3",
        "df_calc_viz_4",
        "df_visualized",
        "df_visualized_1",
        "df_visualized_2",
        "df_visualized_3",
        "df_visualized_4",
        "report_output",
    }
)

# The seven safe normalization operations the compiler emits for a
# ``filter_prep`` step. The order matches the spec wording so audit logs and
# downstream tests can rely on it.
SAFE_FILTER_PREP_OPERATIONS: List[str] = [
    "trim_whitespace",
    "normalize_column_names",
    "normalize_empty_strings",
    "safe_numeric_conversion",
    "safe_currency_conversion",
    "safe_date_detection",
    "categorical_value_normalization",
]


# ---------------------------------------------------------------------------
# Contract validation wrappers
# ---------------------------------------------------------------------------


def _validated_operator(operator: str) -> str:
    """Validate operator against contract registry, raise SemanticCompilationError on failure."""
    try:
        return validate_operator(operator).value
    except InvalidOperatorError as e:
        raise SemanticCompilationError(str(e)) from e


def _validated_action_kind(kind: str) -> str:
    """Validate action kind against contract registry, raise SemanticCompilationError on failure."""
    try:
        return validate_action_kind(kind).value
    except InvalidActionKindError as e:
        raise SemanticCompilationError(str(e)) from e


def _resolved_semantic_operator(relation_operator: str) -> str:
    """Resolve a semantic relation operator to its canonical operator via the contract registry.

    Wraps UnmappedSemanticTypeError as SemanticCompilationError so callers
    see a single exception type for all compilation failures.
    """
    try:
        return resolve_semantic_operator(relation_operator).value
    except UnmappedSemanticTypeError as e:
        raise SemanticCompilationError(str(e)) from e


def _resolved_semantic_operation_type(operation_type: str) -> str:
    """Resolve a semantic operation type to its canonical action kind via the contract registry.

    Wraps UnmappedSemanticTypeError as SemanticCompilationError so callers
    see a single exception type for all compilation failures.
    """
    try:
        return resolve_semantic_operation_type(operation_type).value
    except UnmappedSemanticTypeError as e:
        raise SemanticCompilationError(str(e)) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_plan_for_flag(intent: PlanIntent) -> None:
    """Raise ``ValueError`` when any ``needs_X`` flag lacks its matching plan.

    Requirement 2.13 requires the compiler to refuse intents that toggle a
    capability flag without supplying the matching plan, for every
    ``X ∈ {cleaning, filtering, calculation, visualization}``. The error
    message names the missing field so the orchestrator's quarantine reason
    is precise.
    """
    missing: List[str] = []
    if intent.needs_cleaning and intent.cleaning_plan is None:
        missing.append("cleaning_plan")
    if intent.needs_filtering and intent.filter_plan is None:
        missing.append("filter_plan")
    if intent.needs_calculation and intent.calculation_plan is None:
        missing.append("calculation_plan")
    if intent.needs_visualization and intent.visualization_plan is None:
        missing.append("visualization_plan")

    if missing:
        # Surface the first missing field in the message; all four checks are
        # deterministic so the order is stable across runs. The wording
        # ``needs_X is true but X_plan is missing`` names both the offending
        # flag and the missing field so the orchestrator's quarantine reason
        # is fully explanatory.
        field = missing[0]
        flag = "needs_" + field.replace("_plan", "")
        raise ValueError(
            f"{flag} is true but {field} is missing"
        )


def build_reporting_params(
    intent: PlanIntent,
    output_dir: str,
    file_prefix: str,
) -> Dict[str, Any]:
    """Build the ``params`` dict for the trailing ``reporting_agent`` step.

    Kept as a small helper so future enrichments (e.g. propagating reporting
    hints, audit-log markers) have a single attachment point that does not
    require touching the main compile function. The shape matches
    ``ReportingAgentParams`` (Requirement 10.4): a nested ``plan`` produced
    by ``ReportingOperationPlan`` plus the writer's optional output
    locators.
    """
    return {
        "plan": {
            "output_format": intent.output_format,
            "title": intent.reporting_title,
            "sheet_name": intent.sheet_name,
        },
        "output_dir": output_dir,
        "file_prefix": file_prefix,
    }


def _assert_canonical_output_keys(steps: List[PlanStep]) -> None:
    """Defensive post-condition: every emitted ``output_key`` is canonical.

    Requirement 2.14 forbids the compiler from publishing dataframes under
    any key outside ``CANONICAL_OUTPUT_KEYS``. The validator (sibling task
    8.1) re-checks this independently, but a compile-time assertion keeps
    the contract local to this function and surfaces regressions quickly.
    """
    for step in steps:
        if step.output_key is not None and not _is_canonical_output_key(step.output_key):
            raise ValueError(
                f"Compiler emitted non-canonical output_key {step.output_key!r} "
                f"on step {step.step_id!r}; allowed keys: "
                f"{sorted(CANONICAL_OUTPUT_KEYS)}"
            )


def _assert_filter_input_from(steps: List[PlanStep]) -> None:
    """Defensive post-condition: every ``filter_agent`` step reads from
    ``df_cleaned`` or ``df_filter_prepared`` and never from ``df_ingested``.

    Requirement 2.10 forbids the compiler from routing the raw ingestion
    output into the filter agent. The validator re-checks this rule, but a
    compile-time assertion makes accidental regressions in this module fail
    loudly during compilation rather than at validation time.
    """
    for step in steps:
        if step.agent != "filter_agent":
            continue
        if step.input_from == ["df_ingested"]:
            raise ValueError(
                f"Compiler routed raw df_ingested into filter step "
                f"{step.step_id!r}; this is forbidden by Requirement 2.10."
            )
        if step.input_from not in (["df_cleaned"], ["df_filter_prepared"]):
            raise ValueError(
                f"filter_agent step {step.step_id!r} has input_from="
                f"{step.input_from!r}; expected ['df_cleaned'] or "
                f"['df_filter_prepared']."
            )


def _is_canonical_output_key(output_key: str) -> bool:
    """Return True when *output_key* matches the compiler's allowed families."""
    if output_key in CANONICAL_OUTPUT_KEYS:
        return True
    if output_key.startswith("df_calc_viz_") and output_key[len("df_calc_viz_"):].isdigit():
        return True
    if output_key.startswith("df_visualized_") and output_key[len("df_visualized_"):].isdigit():
        return True
    return False


def _chart_requires_visualization_calculation(chart: ChartSpec) -> bool:
    """Return True when *chart* needs a chart-local calculation branch."""
    return bool(chart.group_by)


def _build_visualization_calc_operation(chart: ChartSpec) -> dict[str, Any]:
    """Build the calculation operation that prepares a chart-local dataframe."""
    aggregation = chart.aggregation or "count"
    op_type = {
        "count": "group_count",
        "sum": "group_sum",
        "mean": "group_mean",
    }.get(aggregation, "group_count")
    measure_column = chart.measure
    if aggregation == "count":
        measure_column = None
    elif not measure_column:
        measure_column = chart.y
    return {
        "type": op_type,
        "column": measure_column,
        "group_by": list(chart.group_by or []),
        "output_column": chart.output_field or chart.y,
    }


def _normalize_chart_for_compilation(chart: ChartSpec) -> ChartSpec:
    """Normalize known-safe chart redundancies before compilation.

    Count aggregations are dimension-only calculations; any measure copied from
    the grouping field is redundant and can produce invalid ``OperationResult``
    contracts downstream.
    """
    aggregation = chart.aggregation or "count"
    if aggregation != "count":
        return chart

    normalized_group_by = list(chart.group_by or [])
    normalized_series = chart.series
    if len(normalized_group_by) == 1 and normalized_series == normalized_group_by[0]:
        normalized_series = None

    return chart.model_copy(
        update={
            "measure": None,
            "series": normalized_series,
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_intent_to_plan(
    intent: PlanIntent,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    file_prefix: str,
) -> ExecutionPlan:
    """Deterministically translate a validated ``PlanIntent`` into an
    ``ExecutionPlan`` whose shape obeys every clause of Requirement 2.

    The function is pure: identical inputs produce identical
    ``ExecutionPlan`` objects (modulo Python's dict ordering, which Pydantic
    preserves for ``model_dump`` here). All branches consult
    :func:`get_enable_visualization` rather than reading the environment
    directly so tests can monkeypatch the flag deterministically.

    See :mod:`finflow_agent.planning.compiler` module docstring for the full
    contract.
    """
    # ------------------------------------------------------------------
    # 0. Reject intents whose flags are inconsistent with their plans.
    #    Done first so the user sees the most actionable error when several
    #    issues are present (Requirement 2.13).
    # ------------------------------------------------------------------
    _require_plan_for_flag(intent)

    # PDF is not a supported output_format. PlanIntent's Literal type already
    # forbids it at validation time, but the orchestrator can mutate the
    # field after the fact (e.g. to surface a clearer error to the LLM), so
    # we keep an explicit guard here.
    if intent.output_format == "pdf":
        raise ValueError("PDF output format is not allowed.")

    steps: List[PlanStep] = []
    last_df_key: str = "df_ingested"

    # ------------------------------------------------------------------
    # 1. Ingestion is always the first step (Requirement 2.1).
    # ------------------------------------------------------------------
    steps.append(
        PlanStep(
            step_id="ingest",
            agent="ingestion_agent",
            params={
                "resolved_file_path": resolved_file_path,
                "file_type": file_type,
            },
            depends_on=[],
            input_from=[],
            output_key="df_ingested",
        )
    )

    # ------------------------------------------------------------------
    # 2. Filter pipeline branch.
    #
    #    Three mutually exclusive sub-cases, in priority order:
    #      (a) needs_filtering AND needs_cleaning  -> ingest, clean, filter
    #      (b) needs_filtering AND NOT needs_cleaning -> ingest, filter_prep, filter
    #      (c) NOT needs_filtering AND needs_cleaning -> ingest, clean
    #    The two filtering sub-cases share the trailing ``filter_agent``
    #    step but differ in which preparation step feeds it.
    # ------------------------------------------------------------------
    if intent.needs_filtering and intent.filter_plan is not None:
        if intent.needs_cleaning and intent.cleaning_plan is not None:
            # 2a. Full cleaning requested; emit a destructive cleaning_agent
            #     step in ``mode="clean"`` (Requirement 2.2).
            steps.append(
                PlanStep(
                    step_id="clean",
                    agent="cleaning_agent",
                    params={
                        "plan": intent.cleaning_plan.model_dump(),
                        "mode": "clean",
                    },
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_cleaned",
                )
            )
            filter_input = "df_cleaned"
            filter_depends_on = "clean"
            last_df_key = "df_cleaned"
        else:
            # 2b. Filtering without cleaning; insert a non-destructive
            #     filter_prep step (Requirements 2.6, 2.7). The step is a
            #     cleaning_agent invocation with ``mode="filter_prep"`` and
            #     ONLY the seven safe operations enumerated in
            #     ``SAFE_FILTER_PREP_OPERATIONS``. ``params`` deliberately
            #     omits ``plan`` per the task contract; the cleaning_agent
            #     dispatch (sibling task 5.1) handles the ``filter_prep``
            #     mode using the operations list.
            steps.append(
                PlanStep(
                    step_id="filter_prep",
                    agent="cleaning_agent",
                    params={
                        "mode": "filter_prep",
                        "operations": list(SAFE_FILTER_PREP_OPERATIONS),
                    },
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_filter_prepared",
                )
            )
            filter_input = "df_filter_prepared"
            filter_depends_on = "filter_prep"
            last_df_key = "df_filter_prepared"

        # The filter step itself. Requirement 2.10: ``input_from`` must be
        # exactly one of ``["df_cleaned"]`` or ``["df_filter_prepared"]``,
        # never ``["df_ingested"]``. The post-condition assertion below
        # re-checks this.
        steps.append(
            PlanStep(
                step_id="filter",
                agent="filter_agent",
                params={"plan": intent.filter_plan.model_dump()},
                depends_on=[filter_depends_on],
                input_from=[filter_input],
                output_key="df_filtered",
            )
        )
        last_df_key = "df_filtered"

    elif intent.needs_cleaning and intent.cleaning_plan is not None:
        # 2c. Clean-only branch (no filtering requested).
        steps.append(
            PlanStep(
                step_id="clean",
                agent="cleaning_agent",
                params={
                    "plan": intent.cleaning_plan.model_dump(),
                    "mode": "clean",
                },
                depends_on=["ingest"],
                input_from=["df_ingested"],
                output_key="df_cleaned",
            )
        )
        last_df_key = "df_cleaned"

    # ------------------------------------------------------------------
    # 2.5. Calculation step (after filter/clean, before visualization).
    # ------------------------------------------------------------------
    if intent.needs_calculation and intent.calculation_plan is not None:
        steps.append(
            PlanStep(
                step_id="calculate",
                agent="calculation_agent",
                params={
                    "operations": [
                        op.model_dump() if hasattr(op, "model_dump") else op
                        for op in intent.calculation_plan.operations
                    ],
                },
                depends_on=[steps[-1].step_id],
                input_from=[last_df_key],
                output_key="df_calculated",
            )
        )
        last_df_key = "df_calculated"

    # ------------------------------------------------------------------
    # 3. Visualization (scaffolded; gated by ENABLE_VISUALIZATION).
    #
    #    Only entered when the intent both requested visualization and
    #    supplied a plan; the missing-plan branch was already raised in
    #    step 0. The flag is read via ``get_enable_visualization`` so test
    #    fixtures can monkeypatch it deterministically.
    #
    #    Shared transformations (clean/filter/calculate) advance the global
    #    dataframe cursor. Visualization-specific aggregations MUST branch
    #    from that shared source without mutating it for unrelated charts.
    # ------------------------------------------------------------------
    if intent.needs_visualization and intent.visualization_plan is not None:
        if not get_enable_visualization():
            raise VisualizationDisabledError(
                VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE
            )

        charts = list(intent.visualization_plan.charts)
        visualization_source_key = last_df_key
        visualization_depends_on = [steps[-1].step_id]

        if len(charts) == 1:
            chart = _normalize_chart_for_compilation(charts[0])
            viz_input_key = visualization_source_key
            viz_depends_on = list(visualization_depends_on)

            if _chart_requires_visualization_calculation(chart):
                steps.append(
                    PlanStep(
                        step_id="calc_viz",
                        agent="calculation_agent",
                        params={"operations": [_build_visualization_calc_operation(chart)]},
                        depends_on=list(visualization_depends_on),
                        input_from=[visualization_source_key],
                        output_key="df_calc_viz",
                    )
                )
                viz_input_key = "df_calc_viz"
                viz_depends_on = ["calc_viz"]

            steps.append(
                PlanStep(
                    step_id="visualize",
                    agent="visualization_agent",
                    params={
                        "plan": {"charts": [chart.model_dump()]},
                        "chart_descriptions": [chart.title or chart.x or ""],
                    },
                    depends_on=viz_depends_on,
                    input_from=[viz_input_key],
                    output_key="df_visualized",
                )
            )
            last_df_key = "df_visualized"
        else:
            visualization_output_keys: List[str] = []
            visualization_step_ids: List[str] = []

            for index, chart in enumerate(charts, start=1):
                chart = _normalize_chart_for_compilation(chart)
                viz_input_key = visualization_source_key
                viz_depends_on = list(visualization_depends_on)

                if _chart_requires_visualization_calculation(chart):
                    calc_step_id = f"calc_viz_{index}"
                    calc_output_key = f"df_calc_viz_{index}"
                    steps.append(
                        PlanStep(
                            step_id=calc_step_id,
                            agent="calculation_agent",
                            params={"operations": [_build_visualization_calc_operation(chart)]},
                            depends_on=list(visualization_depends_on),
                            input_from=[visualization_source_key],
                            output_key=calc_output_key,
                        )
                    )
                    viz_input_key = calc_output_key
                    viz_depends_on = [calc_step_id]

                viz_step_id = f"visualize_{index}"
                viz_output_key = f"df_visualized_{index}"
                steps.append(
                    PlanStep(
                        step_id=viz_step_id,
                        agent="visualization_agent",
                        params={
                            "plan": {"charts": [chart.model_dump()]},
                            "chart_descriptions": [chart.title or chart.x or ""],
                        },
                        depends_on=viz_depends_on,
                        input_from=[viz_input_key],
                        output_key=viz_output_key,
                    )
                )
                visualization_step_ids.append(viz_step_id)
                visualization_output_keys.append(viz_output_key)

            last_df_key = visualization_source_key
            steps.append(
                PlanStep(
                    step_id="report",
                    agent="reporting_agent",
                    params=build_reporting_params(intent, output_dir, file_prefix),
                    depends_on=visualization_step_ids,
                    input_from=[visualization_source_key, *visualization_output_keys],
                    output_key="report_output",
                )
            )

            plan = ExecutionPlan(steps=steps)
            assert plan.steps[0].agent == "ingestion_agent", (
                "Compiler post-condition violated: first step must be ingestion_agent"
            )
            assert plan.steps[-1].agent == "reporting_agent", (
                "Compiler post-condition violated: last step must be reporting_agent"
            )
            _assert_canonical_output_keys(plan.steps)
            _assert_filter_input_from(plan.steps)
            return plan

    # ------------------------------------------------------------------
    # 4. Reporting is always the last step (Requirement 2.1).
    # ------------------------------------------------------------------
    steps.append(
        PlanStep(
            step_id="report",
            agent="reporting_agent",
            params=build_reporting_params(intent, output_dir, file_prefix),
            depends_on=[steps[-1].step_id],
            input_from=[last_df_key],
            output_key="report_output",
        )
    )

    plan = ExecutionPlan(steps=steps)

    # ------------------------------------------------------------------
    # 5. Sanity post-conditions. The validator re-checks all of these but
    #    asserting them here pins the contract local to the compiler and
    #    surfaces regressions in this module immediately.
    # ------------------------------------------------------------------
    assert plan.steps[0].agent == "ingestion_agent", (
        "Compiler post-condition violated: first step must be ingestion_agent"
    )
    assert plan.steps[-1].agent == "reporting_agent", (
        "Compiler post-condition violated: last step must be reporting_agent"
    )
    _assert_canonical_output_keys(plan.steps)
    _assert_filter_input_from(plan.steps)

    return plan


def compile_canonical_intent(
    intent: CanonicalIntent,
    *,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    artifact_prefix: str,
) -> ExecutionPlan:
    """Compile a validated ``CanonicalIntent`` into an ``ExecutionPlan``.

    This is the production entrypoint for worker-side execution planning.
    It does not accept raw prompts or free-form dicts.
    """
    _validate_canonical_intent(intent)
    plan_intent = _canonical_intent_to_plan_intent(
        intent,
        resolved_file_path=resolved_file_path,
        file_type=file_type,
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
    )
    return compile_intent_to_plan(
        intent=plan_intent,
        resolved_file_path=resolved_file_path,
        file_type=file_type,
        output_dir=output_dir,
        file_prefix=artifact_prefix,
    )


def _validate_canonical_intent(intent: CanonicalIntent) -> None:
    if intent.schema_version != CANONICAL_INTENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported canonical intent schema_version {intent.schema_version!r}; "
            f"expected {CANONICAL_INTENT_SCHEMA_VERSION!r}."
        )
    if intent.output_format not in {"xlsx", "csv", "json", "txt"}:
        raise ValueError(f"Unsupported output_format {intent.output_format!r}.")
    if intent.resolution_status not in {"resolved", "repaired"}:
        raise ValueError(
            f"Canonical intent resolution_status {intent.resolution_status!r} is not executable."
        )
    if not intent.actions:
        raise ValueError("Canonical intent contains no executable actions.")


def _canonical_intent_to_plan_intent(
    intent: CanonicalIntent,
    *,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    artifact_prefix: str,
) -> PlanIntent:
    source_columns = _source_columns_from_intent(intent, resolved_file_path)

    cleaning_plan: CleaningOperationPlan | None = None
    filter_plan: FilterOperationPlan | None = None
    calculation_plan: CalculationOperationPlan | None = None
    needs_cleaning = False
    needs_filtering = False
    needs_calculation = False
    needs_visualization = False

    selected_columns: list[str] = []
    filter_conditions: list[FilterCondition] = []
    filter_logic = "and"
    filter_limit: int | None = None
    drop_columns_seen = False
    calculation_operations: list = []
    visualize_chart_type: str | None = None
    visualize_group_by: list[str] | None = None
    visualize_aggregation: str | None = None
    visualize_measure: str | None = None
    visualize_output_field: str | None = None
    visualize_x: str | None = None
    visualize_series: str | None = None
    visualize_bar_mode: str | None = None
    visualize_show_legend: bool | None = None
    visualize_show_data_labels: bool | None = None
    visualize_title: str | None = None
    visualize_x_axis_title: str | None = None
    visualize_y_axis_title: str | None = None
    visualize_intents: list[dict] = []

    for action in intent.actions:
        # Resolve the semantic operation type to a canonical action kind via
        # the contract registry.  This ensures the compiler uses the single
        # source-of-truth mapping (Requirements 5.2, 6.1, 7.4).
        _resolved_semantic_operation_type(action.kind)

        if isinstance(action, CleanIntent):
            cleaning_plan = _build_cleaning_plan(action)
            needs_cleaning = True
            continue

        if isinstance(action, ProjectColumnsIntent):
            selected_columns = _resolve_requested_columns(action.requested_fields)
            needs_filtering = True
            continue

        if isinstance(action, DropColumnsIntent):
            drop_columns_seen = True
            if not source_columns:
                raise ValueError("drop_columns canonical intent requires source columns in the dataframe profile.")
            dropped = _resolve_requested_columns(action.requested_fields)
            selected_columns = [column for column in source_columns if column not in set(dropped)]
            needs_filtering = True
            continue

        if isinstance(action, FilterRowsIntent):
            filter_logic = action.logic
            needs_filtering = True
            filter_conditions.extend(_canonical_filter_conditions(action.conditions, action.mode))
            continue

        if isinstance(action, LimitRowsIntent):
            filter_limit = max(0, int(action.limit))
            needs_filtering = True
            continue

        if isinstance(action, SortRowsIntent):
            # Sort is not yet compiled into the execution plan — skip gracefully
            continue
        if isinstance(action, CalculateIntent):
            ops = _build_calculation_operations(action)
            if ops:
                needs_calculation = True
                calculation_operations = ops
            continue
        if isinstance(action, VisualizeIntent):
            needs_visualization = True
            # Collect all visualization intents for multi-chart support
            visualize_intents.append({
                "chart_type": action.chart_type,
                "group_by": action.group_by,
                "aggregation": action.aggregation,
                "measure": action.measure,
                "output_field": action.output_field,
            })
            # Keep last one as fallback for single-chart variables
            visualize_chart_type = action.chart_type
            visualize_group_by = action.group_by
            visualize_aggregation = action.aggregation
            visualize_measure = action.measure
            visualize_output_field = action.output_field
            visualize_x = action.x
            visualize_series = action.series
            visualize_bar_mode = action.bar_mode
            visualize_show_legend = action.show_legend
            visualize_show_data_labels = action.show_data_labels
            visualize_title = action.title
            visualize_x_axis_title = action.x_axis_title
            visualize_y_axis_title = action.y_axis_title
            visualize_intents[-1].update(
                {
                    "x": action.x,
                    "series": action.series,
                    "bar_mode": action.bar_mode,
                    "show_legend": action.show_legend,
                    "show_data_labels": action.show_data_labels,
                    "title": action.title,
                    "x_axis_title": action.x_axis_title,
                    "y_axis_title": action.y_axis_title,
                }
            )
            continue
        if isinstance(action, ReportIntent):
            continue

        raise ValueError(f"Unsupported canonical intent action kind: {getattr(action, 'kind', type(action).__name__)}")

    if drop_columns_seen and not source_columns:
        raise ValueError("drop_columns canonical intent cannot be compiled without source columns.")

    if selected_columns or filter_conditions or filter_limit is not None:
        filter_plan = FilterOperationPlan(
            conditions=filter_conditions,
            logic=filter_logic,
            select_columns=selected_columns or None,
            limit=filter_limit,
        )
        needs_filtering = True

    if calculation_operations:
        calculation_plan = CalculationOperationPlan(
            operations=[
                CalculationOperation(**op) if isinstance(op, dict) else op
                for op in calculation_operations
            ]
        )

    # Build visualization plan from extracted chart fields
    visualization_plan: VisualizationOperationPlan | None = None
    if needs_visualization:
        chart_type_map = {"pie": "pie", "bar": "bar", "line": "line", "scatter": "scatter", "histogram": "bar", "area": "area"}
        charts_list: List[ChartSpec] = []

        # Build one ChartSpec per visualize intent
        intents_to_process = visualize_intents if visualize_intents else [{
            "chart_type": visualize_chart_type,
            "group_by": visualize_group_by,
            "aggregation": visualize_aggregation,
            "measure": visualize_measure,
            "output_field": visualize_output_field,
            "x": visualize_x,
            "series": visualize_series,
            "bar_mode": visualize_bar_mode,
            "show_legend": visualize_show_legend,
            "show_data_labels": visualize_show_data_labels,
            "title": visualize_title,
            "x_axis_title": visualize_x_axis_title,
            "y_axis_title": visualize_y_axis_title,
        }]

        for viz_intent in intents_to_process:
            chart_type = viz_intent.get("chart_type") or "bar"
            resolved_chart_type = chart_type_map.get(chart_type.lower(), "bar")
            viz_group_by = viz_intent.get("group_by")
            viz_measure = viz_intent.get("measure")
            viz_aggregation = viz_intent.get("aggregation")
            viz_output_field = viz_intent.get("output_field")
            viz_x = viz_intent.get("x")
            viz_series = viz_intent.get("series")
            viz_bar_mode = viz_intent.get("bar_mode")

            # Determine x and y fields
            x_field = ""
            y_field = ""
            agg_type = viz_aggregation or "count"

            if viz_x:
                x_field = viz_x
            elif viz_group_by and isinstance(viz_group_by, list) and viz_group_by:
                x_field = viz_group_by[0]
            if viz_measure:
                y_field = viz_measure

            # Derive from calculation operations
            if not x_field and calculation_operations:
                for calc_op in calculation_operations:
                    op_dict = calc_op if isinstance(calc_op, dict) else {}
                    gb = op_dict.get("group_by")
                    if gb and isinstance(gb, list) and gb:
                        x_field = gb[0]
                        if not y_field:
                            op_t = op_dict.get("type", "")
                            y_field = op_dict.get("output_column") or f"{op_t.replace('group_', '')}_{op_dict.get('column', '')}"
                            agg_type = op_t.replace("group_", "") if "group_" in op_t else op_t
                        break

            if not y_field:
                y_field = viz_output_field or "record_count"

            # Fallback x_field from description or source columns
            if not x_field and source_columns:
                # Try to match chart description against column names
                viz_description = viz_intent.get("description", "")
                if viz_description:
                    desc_lower = viz_description.lower().replace("_", " ")
                    for col in source_columns:
                        col_normalized = col.lower().replace("_", " ")
                        if col_normalized in desc_lower or desc_lower in col_normalized:
                            x_field = col
                            break
                    # Partial word match
                    if not x_field:
                        desc_words = [w for w in desc_lower.split() if len(w) > 3]
                        for col in source_columns:
                            col_lower = col.lower().replace("_", " ")
                            if any(w in col_lower for w in desc_words):
                                x_field = col
                                break

                # Generic fallback: first categorical keyword match
                if not x_field:
                    for col in source_columns:
                        col_lower = col.lower()
                        if any(kw in col_lower for kw in ["gender", "category", "type", "status", "class", "department", "group", "education"]):
                            x_field = col
                            break
                if not x_field:
                    x_field = source_columns[0]

            # Ensure group_by for aggregation charts
            effective_group_by = viz_group_by
            if not effective_group_by and x_field and resolved_chart_type in ("pie", "bar"):
                effective_group_by = [x_field]
            if viz_series:
                if effective_group_by is None:
                    effective_group_by = []
                if x_field and not effective_group_by:
                    effective_group_by.append(x_field)
                if viz_series not in effective_group_by:
                    effective_group_by.append(viz_series)
            if effective_group_by and len(effective_group_by) > 2:
                raise ValueError(
                    "TOO_MANY_VISUALIZATION_DIMENSIONS: bar charts support at most two grouping dimensions."
                )

            effective_measure = viz_measure
            if not effective_measure and calculation_operations:
                for calc_op in calculation_operations:
                    op_dict = calc_op if isinstance(calc_op, dict) else {}
                    if op_dict.get("group_by"):
                        effective_measure = op_dict.get("column")
                        break

            effective_aggregation = agg_type if agg_type != "count" else (viz_aggregation or "count")
            normalized_series = viz_series
            if effective_aggregation == "count":
                effective_measure = None
                if effective_group_by and len(effective_group_by) == 1 and normalized_series == effective_group_by[0]:
                    normalized_series = None

            try:
                chart_title = (
                    viz_intent.get("title")
                    or viz_intent.get("description")
                    or f"{resolved_chart_type.capitalize()} Chart"
                )
                charts_list.append(ChartSpec(
                    type=resolved_chart_type,
                    x=x_field,
                    y=y_field,
                    title=chart_title,
                    group_by=effective_group_by,
                    series=normalized_series or (effective_group_by[1] if effective_group_by and len(effective_group_by) > 1 else None),
                    measure=effective_measure,
                    aggregation=effective_aggregation,
                    output_field=y_field,
                    bar_mode=viz_bar_mode or ("grouped" if resolved_chart_type == "bar" and effective_group_by and len(effective_group_by) > 1 else None),
                    show_legend=viz_intent.get("show_legend"),
                    show_data_labels=viz_intent.get("show_data_labels"),
                    x_axis_title=viz_intent.get("x_axis_title"),
                    y_axis_title=viz_intent.get("y_axis_title"),
                ))
            except Exception:
                continue

        if charts_list:
            visualization_plan = VisualizationOperationPlan(charts=charts_list)
        else:
            needs_visualization = False
            visualization_plan = None

    return PlanIntent(
        needs_cleaning=needs_cleaning,
        needs_filtering=needs_filtering,
        needs_calculation=needs_calculation,
        needs_visualization=needs_visualization,
        output_format=intent.output_format,
        cleaning_plan=cleaning_plan,
        filter_plan=filter_plan,
        calculation_plan=calculation_plan,
        visualization_plan=visualization_plan,
        reporting_title=intent.decision or None,
        sheet_name=None,
    )


def _build_calculation_operations(action: CalculateIntent) -> list:
    """Build calculation operations from a CalculateIntent action."""
    operations = []
    for op in action.operations:
        if isinstance(op, dict):
            operations.append(op)
        elif isinstance(op, str):
            # Legacy format: just a string description — skip
            continue
        else:
            operations.append(op)
    return operations


def _source_columns_from_intent(intent: CanonicalIntent, resolved_file_path: str) -> list[str]:
    profile_columns = intent.dataframe_profile.get("source_columns")
    if not isinstance(profile_columns, list):
        # Fallback: some pipelines emit "columns" instead of "source_columns"
        profile_columns = intent.dataframe_profile.get("columns")
    if isinstance(profile_columns, list):
        columns = [str(column).strip() for column in profile_columns if str(column).strip()]
        if columns:
            return columns

    path = Path(str(resolved_file_path))
    if not path.exists():
        return []

    extension = path.suffix.lower()
    try:
        if extension in {".csv", ".tsv"}:
            frame = pd.read_csv(path, nrows=0, sep="\t" if extension == ".tsv" else ",")
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, nrows=0)
        else:
            return []
        return [str(column).strip() for column in frame.columns if str(column).strip()]
    except Exception:
        return []


def _derive_group_by_from_profile(intent: CanonicalIntent) -> list[str] | None:
    """Derive a default group_by column from the dataframe profile and prompt.

    Strategy:
    1. First, try to find a column name that appears in the user's prompt
       (e.g., "gender" in "show male to female ratio as a pie chart").
    2. If no prompt match, look for known low-cardinality categorical columns
       (gender, sex, status, type, etc.) preferring exact matches over suffix matches.
    """
    source_columns = intent.dataframe_profile.get("source_columns", [])
    if not source_columns:
        return None

    prompt = (intent.original_prompt or "").lower()

    # Strategy 1: Find columns explicitly mentioned in the prompt
    # Also check common synonyms (male/female → gender, etc.)
    prompt_column_hints = {
        "male": "gender", "female": "gender", "gender": "gender",
        "sex": "sex", "men": "gender", "women": "gender",
    }

    # Direct column name match in prompt
    for col in source_columns:
        col_lower = str(col).lower().strip()
        if col_lower in prompt:
            return [str(col).strip()]

    # Synonym-based match
    for keyword, target_col in prompt_column_hints.items():
        if keyword in prompt:
            # Find the actual column that matches the target
            for col in source_columns:
                if str(col).lower().strip() == target_col:
                    return [str(col).strip()]

    # Strategy 2: Known low-cardinality categorical columns (exact match first)
    # Prioritize columns that are typically binary/few categories
    high_priority = {"gender", "sex", "status", "type", "class", "tier", "level"}
    low_priority = {"category", "region", "segment", "group", "department",
                    "country", "city", "state", "brand", "product", "channel"}

    for col in source_columns:
        col_lower = str(col).lower().strip()
        if col_lower in high_priority:
            return [str(col).strip()]

    for col in source_columns:
        col_lower = str(col).lower().strip()
        if col_lower in low_priority:
            return [str(col).strip()]

    # Strategy 3: Suffix/prefix patterns (less reliable)
    all_hints = high_priority | low_priority
    for col in source_columns:
        col_lower = str(col).lower().strip()
        for hint in high_priority:
            if col_lower.endswith(f"_{hint}") or col_lower.startswith(f"{hint}_"):
                return [str(col).strip()]

    return None


def _resolve_requested_columns(fields: list[UnresolvedColumnReference]) -> list[str]:
    resolved: list[str] = []
    for field in fields:
        resolved.extend(_resolve_field_columns(field))
    return list(dict.fromkeys(resolved))


def _resolve_field_columns(field: UnresolvedColumnReference) -> list[str]:
    if field.resolved_columns:
        columns = [str(column).strip() for column in field.resolved_columns if str(column).strip()]
        if columns:
            return columns

    column = str(field.resolved_column or "").strip()
    if column:
        return [column]

    candidates = [str(column).strip() for column in field.candidate_columns if str(column).strip()]
    detail = f"selection_mode={field.selection_mode!r}"
    if candidates:
        detail += f", candidates={candidates!r}"
    raise ValueError(f"Unresolved canonical column reference: {field.raw_reference!r} ({detail})")


def _build_cleaning_plan(action: CleanIntent) -> CleaningOperationPlan:
    if not action.operations:
        return CleaningOperationPlan(
            operations=[
                TrimWhitespaceOperation(columns="__all_string_columns__"),
                NormalizeColumnNamesOperation(style="snake_case"),
                DropDuplicatesOperation(subset=None, keep="first"),
            ]
        )

    operations: list[Any] = []
    for operation in action.operations:
        name = operation.name.strip()
        if name == "trim_whitespace":
            operations.append(TrimWhitespaceOperation(columns="__all_string_columns__"))
        elif name == "normalize_column_names":
            operations.append(NormalizeColumnNamesOperation(style="snake_case"))
        elif name in ("drop_duplicates", "deduplicate"):
            operations.append(DropDuplicatesOperation(subset=None, keep="first"))
        elif name == "drop_nulls":
            columns = operation.parameters.get("columns") if isinstance(operation.parameters, dict) else None
            how = str(operation.parameters.get("how", "any")).strip().lower() if isinstance(operation.parameters, dict) else "any"
            if columns is not None and not isinstance(columns, list):
                raise ValueError("drop_nulls cleaning operation requires columns to be null or a list of column names.")
            operations.append(DropNullsOperation(columns=columns, how=how if how in {"any", "all"} else "any"))
        elif name == "remove_empty_rows":
            operations.append(RemoveEmptyRowsOperation())
        elif name == "absolute_value":
            columns = "__all_numeric_columns__"
            if isinstance(operation.parameters, dict) and operation.parameters.get("columns"):
                columns = operation.parameters["columns"]
            operations.append(AbsoluteValueOperation(columns=columns))
        elif name == "fill_nulls":
            params = operation.parameters if isinstance(operation.parameters, dict) else {}
            operations.append(FillNullsOperation(
                column=params.get("column", "__all__"),
                strategy=params.get("strategy", "constant"),
                value=params.get("value", 0),
            ))
        elif name == "normalize_date":
            operations.append(NormalizeDateOperation(
                column=operation.parameters.get("column", "__all__") if isinstance(operation.parameters, dict) else "__all__",
                target_format="%Y-%m-%d",
            ))
        elif name == "normalize_text_case":
            operations.append(NormalizeTextCaseOperation(
                columns="__all_string_columns__",
                case="lower",
            ))
        elif name == "strip_currency_symbols":
            operations.append(StripCurrencySymbolsOperation(
                columns="__all_string_columns__",
            ))
        elif name == "remove_commas_from_numbers":
            operations.append(RemoveCommasFromNumbersOperation(
                columns="__all_string_columns__",
            ))
        elif name == "normalize_categorical_values":
            # Pass through to the handler — it operates on all string columns by default
            continue  # No schema model needed — handled dynamically by the handler
        else:
            # Skip unknown operations gracefully instead of crashing
            continue
    return CleaningOperationPlan(operations=operations)


def _canonical_filter_conditions(
    conditions: list[FilterCondition],
    mode: str,
) -> list[FilterCondition]:
    if mode == "keep":
        result = []
        for condition in conditions:
            # Skip conditions with unresolved generic field references
            if not _can_resolve_field(condition.field):
                continue
            # Skip conditions with null value for operators that require one
            if condition.value is None and condition.operator not in ("is_null", "is_not_null"):
                continue
            result.append(
                FilterCondition(
                    column=_resolve_single_grounded_column(condition.field),
                    operator=_resolved_semantic_operator(condition.operator),
                    value=condition.value,
                )
            )
        return result

    inverted: list[FilterCondition] = []
    for condition in conditions:
        if not _can_resolve_field(condition.field):
            continue
        if condition.value is None and condition.operator not in ("is_null", "is_not_null"):
            continue
        operator = _invert_operator(condition.operator)
        inverted.append(
            FilterCondition(
                column=_resolve_single_grounded_column(condition.field),
                operator=_resolved_semantic_operator(operator),
                value=condition.value,
            )
        )
    return inverted


def _can_resolve_field(field) -> bool:
    """Check if a field reference can be resolved to a column name."""
    if hasattr(field, "resolved_column") and field.resolved_column:
        return True
    if hasattr(field, "resolved_columns") and field.resolved_columns:
        return True
    # Generic/unresolved references cannot be compiled
    raw_ref = str(getattr(field, "raw_reference", "") or "").strip().lower()
    if raw_ref in {"generic", "", "unknown", "null", "none"}:
        return False
    # If resolution_method is generic_reference with no resolved column, skip
    method = str(getattr(field, "resolution_method", "") or "").strip().lower()
    if method == "generic_reference":
        return False
    return False


def _resolve_single_grounded_column(field: UnresolvedColumnReference) -> str:
    column = str(field.resolved_column or "").strip()
    if column:
        return column
    if field.resolved_columns:
        raise ValueError(
            f"Canonical filter field {field.raw_reference!r} resolved to multiple columns {field.resolved_columns!r}; "
            "filter conditions require a single grounded column."
        )
    raise ValueError(f"Unresolved canonical filter field: {field.raw_reference!r}")


def _invert_operator(operator: str) -> str:
    mapping = {
        "eq": "neq",
        "neq": "eq",
        "gt": "lte",
        "gte": "lt",
        "lt": "gte",
        "lte": "gt",
        "contains": "not_contains",
    }
    if operator not in mapping:
        raise ValueError(f"Unsupported operator for drop-mode canonical intent: {operator!r}")
    return mapping[operator]


__all__ = [
    "SemanticCompilationError",
    "VisualizationDisabledError",
    "VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE",
    "CANONICAL_OUTPUT_KEYS",
    "SAFE_FILTER_PREP_OPERATIONS",
    "build_reporting_params",
    "compile_intent_to_plan",
    # --- New refactored pipeline exports (Semantic Grounding Refactor) ---
    "CompilerError",
    "ExecutionStep",
    "RefactoredExecutionPlan",
    "Compiler",
]


# ===========================================================================
# NEW: Refactored Compiler (Semantic Grounding Refactor)
#
# This section implements the deterministic Compiler contract from the
# semantic-grounding-refactor spec. It accepts only the new CanonicalIntent
# model (from models/canonical.py), validates all column references are
# resolved (type-level guarantee), and produces an ExecutionPlan.
#
# Requirements: 6.1, 6.4, 11.1, 11.2
# ===========================================================================

from enum import Enum
from typing import Literal as TypingLiteral
from uuid import uuid4 as _uuid4

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.canonical import (
    CanonicalIntent as RefactoredCanonicalIntent,
    ResolvedAction,
    ResolvedFilterAction,
    ResolvedProjectAction,
    ResolvedDropAction,
    ResolvedSortAction,
    ResolvedRenameAction,
)


class CompilerError(Exception):
    """Raised when the Compiler detects a contract violation.

    This covers:
    - Input that is not a CanonicalIntent
    - Column references that are not resolved to physical columns
    - Any attempt to perform grounding or semantic re-interpretation

    Requirements: 11.1, 11.2
    """

    pass


class ActionType(str, Enum):
    """Enumeration of execution step action types."""

    FILTER = "filter"
    PROJECT = "project"
    DROP = "drop"
    SORT = "sort"
    RENAME = "rename"


class ExecutionStep(BaseModel):
    """A single execution step in the execution plan.

    Each step represents one resolved action that the Executor will perform.
    All column references are physical (resolved) column names.

    Requirements: 11.1 - every column reference validated as physical.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    step_id: str = Field(default_factory=lambda: str(_uuid4()))
    action_type: ActionType
    resolved_columns: List[str] = Field(
        ...,
        description="Physical column names involved in this step",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (operators, values, directions, etc.)",
    )


class RefactoredExecutionPlan(BaseModel):
    """An execution plan produced by the refactored Compiler.

    Contains a list of ExecutionSteps plus intent metadata for traceability.
    The Executor walks these steps sequentially — no LLM calls, no grounding,
    no semantic interpretation.

    Requirements: 6.1, 6.4, 11.1
    """

    model_config = ConfigDict(strict=True, frozen=True)

    plan_id: str = Field(default_factory=lambda: str(_uuid4()))
    intent_id: str = Field(
        ..., description="ID of the CanonicalIntent that produced this plan"
    )
    source_draft_id: str = Field(
        ..., description="ID of the original SemanticIntentDraft"
    )
    source_draft_revision: int = Field(
        ..., description="Revision of the original draft"
    )
    schema_version: str = Field(
        default="1.0", description="Plan schema version"
    )
    steps: List[ExecutionStep] = Field(
        ..., description="Ordered list of execution steps"
    )


class Compiler:
    """Deterministic compiler: CanonicalIntent → ExecutionPlan.

    This compiler is the Decision_Owner for execution step selection.
    It accepts only CanonicalIntent objects (from models/canonical.py),
    which by type-level guarantee contain no unresolved active execution
    references. It validates all column references are resolved to
    physical columns and produces an ExecutionPlan.

    Contract:
    - Accepts ONLY CanonicalIntent (type-level guarantee, Req 11.2)
    - Validates every column reference resolved to physical column (Req 11.1)
    - Does NOT perform grounding (Req 6.4)
    - Does NOT make LLM calls (Req 6.4)
    - Does NOT re-interpret semantics (Req 6.4)
    - Produces a RefactoredExecutionPlan

    Requirements: 6.1, 6.4, 11.1, 11.2
    """

    def compile(self, intent: RefactoredCanonicalIntent) -> RefactoredExecutionPlan:
        """Compile a canonical intent into an execution plan.

        Preconditions (type-level guarantees):
        - Input is CanonicalIntent (always resolved)
        - Every column reference resolved to physical column

        Does NOT: perform grounding, make LLM calls, re-interpret semantics.

        Args:
            intent: A fully-resolved CanonicalIntent.

        Returns:
            A RefactoredExecutionPlan with ordered ExecutionSteps.

        Raises:
            CompilerError: If the input is not a CanonicalIntent or any
                column reference is not resolved.
        """
        # Type-level contract enforcement (Req 11.2)
        if not isinstance(intent, RefactoredCanonicalIntent):
            raise CompilerError(
                f"Compiler accepts only CanonicalIntent objects. "
                f"Received: {type(intent).__name__}"
            )

        # Validate resolution_status (redundant due to type literal, but
        # belt-and-suspenders for contract enforcement)
        if intent.resolution_status != "resolved":
            raise CompilerError(
                f"CanonicalIntent must have resolution_status='resolved'. "
                f"Got: {intent.resolution_status!r}"
            )

        # Validate actions are present
        if not intent.actions:
            raise CompilerError(
                "CanonicalIntent contains no actions to compile."
            )

        # Walk resolved actions and compile to execution steps
        steps: List[ExecutionStep] = []
        for action in intent.actions:
            step = self._compile_action(action)
            steps.append(step)

        return RefactoredExecutionPlan(
            intent_id=intent.intent_id,
            source_draft_id=intent.source_draft_id,
            source_draft_revision=intent.source_draft_revision,
            steps=steps,
        )

    def _compile_action(self, action: ResolvedAction) -> ExecutionStep:
        """Compile a single resolved action into an ExecutionStep.

        Validates that all column references within the action are resolved
        to physical column names (non-empty strings).

        Raises CompilerError if any column reference is invalid.
        """
        if isinstance(action, ResolvedFilterAction):
            return self._compile_filter(action)
        elif isinstance(action, ResolvedProjectAction):
            return self._compile_project(action)
        elif isinstance(action, ResolvedDropAction):
            return self._compile_drop(action)
        elif isinstance(action, ResolvedSortAction):
            return self._compile_sort(action)
        elif isinstance(action, ResolvedRenameAction):
            return self._compile_rename(action)
        else:
            raise CompilerError(
                f"Unknown resolved action type: {type(action).__name__}"
            )

    def _compile_filter(self, action: ResolvedFilterAction) -> ExecutionStep:
        """Compile a resolved filter action into an ExecutionStep."""
        columns: List[str] = []
        for predicate in action.predicates:
            col = predicate.get("column")
            if not col or not isinstance(col, str) or not col.strip():
                raise CompilerError(
                    f"Filter predicate has unresolved or empty column reference: "
                    f"{predicate!r}"
                )
            columns.append(str(col))

        self._validate_columns_resolved(columns, "filter")

        return ExecutionStep(
            action_type=ActionType.FILTER,
            resolved_columns=columns,
            params={"predicates": [dict(p) for p in action.predicates]},
        )

    def _compile_project(self, action: ResolvedProjectAction) -> ExecutionStep:
        """Compile a resolved project action into an ExecutionStep."""
        self._validate_columns_resolved(action.columns, "project")

        return ExecutionStep(
            action_type=ActionType.PROJECT,
            resolved_columns=list(action.columns),
            params={},
        )

    def _compile_drop(self, action: ResolvedDropAction) -> ExecutionStep:
        """Compile a resolved drop action into an ExecutionStep."""
        self._validate_columns_resolved(action.columns, "drop")

        return ExecutionStep(
            action_type=ActionType.DROP,
            resolved_columns=list(action.columns),
            params={},
        )

    def _compile_sort(self, action: ResolvedSortAction) -> ExecutionStep:
        """Compile a resolved sort action into an ExecutionStep."""
        self._validate_columns_resolved(action.keys, "sort")

        if len(action.keys) != len(action.directions):
            raise CompilerError(
                f"Sort action has mismatched keys ({len(action.keys)}) "
                f"and directions ({len(action.directions)})."
            )

        return ExecutionStep(
            action_type=ActionType.SORT,
            resolved_columns=list(action.keys),
            params={"directions": list(action.directions)},
        )

    def _compile_rename(self, action: ResolvedRenameAction) -> ExecutionStep:
        """Compile a resolved rename action into an ExecutionStep."""
        source_columns: List[str] = []
        rename_map: Dict[str, str] = {}

        for source_col, new_name in action.mappings:
            if not source_col or not source_col.strip():
                raise CompilerError(
                    f"Rename action has unresolved source column: "
                    f"'{source_col}' -> '{new_name}'"
                )
            if not new_name or not new_name.strip():
                raise CompilerError(
                    f"Rename action has empty target name for column: "
                    f"'{source_col}'"
                )
            source_columns.append(source_col)
            rename_map[source_col] = new_name

        self._validate_columns_resolved(source_columns, "rename")

        return ExecutionStep(
            action_type=ActionType.RENAME,
            resolved_columns=source_columns,
            params={"rename_map": rename_map},
        )

    @staticmethod
    def _validate_columns_resolved(
        columns: List[str], context: str
    ) -> None:
        """Validate that all column names are non-empty resolved physical columns.

        Raises CompilerError if any column is empty or whitespace-only.

        Requirements: 11.1 - every column reference validated as physical.
        """
        for col in columns:
            if not col or not col.strip():
                raise CompilerError(
                    f"Unresolved or empty column reference in {context} action: "
                    f"'{col}'. All columns must be resolved to physical names "
                    f"before compilation."
                )
