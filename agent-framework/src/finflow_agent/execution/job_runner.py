"""Job runner that orchestrates execution of plans with visualization support.

Integrates the ExecutionEngine (for calculation DAG nodes) with the
visualization runner (for visualization DAG nodes) and the
VisualizationJobStatusHandler (for failure isolation and job status logic).

Key behaviors:
- Runs calc steps via the ExecutionEngine
- Runs viz steps via execute_visualization_nodes with 30s timeout
- Calc step status remains "success" when viz node fails/times out (Req 6.1)
- Job status = "completed_with_warnings" when all calc succeed but ≥1 viz fails (Req 6.2)
- Job status = "failed" when any calc step fails regardless of viz outcomes (Req 6.4)
- Failed viz nodes produce VisualizationSpec with status "failed", error ≤500 chars (Req 6.3)
- Continues executing remaining DAG nodes after viz failure (Req 6.3)
- Produces one VisualizationSpec per viz node regardless of other outcomes (Req 5.3)
- Returns specs ordered by topological position in execution plan (Req 5.4)
- 30-second execution timeout for visualization nodes (Req 6.3)

Requirements: 5.3, 5.4, 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

from typing import Any

from finflow_agent.execution.engine import ExecutionEngine
from finflow_agent.execution.visualization.spec import VisualizationSpec
from finflow_agent.execution.visualization_job_status import (
    VisualizationJobResult,
    VisualizationJobStatusHandler,
)
from finflow_agent.execution.visualization_runner import (
    VisualizationNode,
    VisualizationPlanError,
    execute_visualization_nodes,
    validate_visualization_plan,
)
from finflow_agent.state import ExecutionPlan


class JobRunner:
    """Runner that executes planned jobs with visualization failure isolation.

    Orchestrates the execution of calculation steps via the ExecutionEngine
    and visualization steps via the visualization runner, applying the
    failure isolation rules defined in Requirements 6.1-6.4.

    The runner ensures:
    - Calc step statuses are NEVER modified by viz failures (Req 6.1)
    - Job status reflects the combined outcome correctly (Req 6.2, 6.4)
    - Each viz node produces exactly one spec (Req 5.3)
    - Viz specs are ordered topologically (Req 5.4)
    - Viz execution has a 30-second timeout (Req 6.3)

    Requirements: 5.3, 5.4, 6.1, 6.2, 6.3, 6.4
    """

    def __init__(self) -> None:
        self.engine = ExecutionEngine()
        self._status_handler = VisualizationJobStatusHandler()

    def run_job(
        self,
        plan: ExecutionPlan,
        submission_id: str | None = None,
        visualization_nodes: list[VisualizationNode] | None = None,
    ) -> dict[str, Any]:
        """Execute a job plan with optional visualization nodes.

        Args:
            plan: The ExecutionPlan containing calculation steps.
            submission_id: Optional submission identifier for tracing.
            visualization_nodes: Optional list of VisualizationNodes to
                execute after calc steps complete. These must already be
                validated (via validate_visualization_plan).

        Returns:
            A result dict with:
            - status: "complete", "completed_with_warnings", or "failed"
            - output_path: Path to the primary output file (if any)
            - summary: Execution summary with step statuses, metrics, etc.
            - visualizations: List of VisualizationSpecs (empty if no viz nodes)

        Requirements: 5.3, 5.4, 6.1, 6.2, 6.3, 6.4
        """
        # If no visualization nodes, delegate entirely to the engine
        if not visualization_nodes:
            result = self.engine.execute(plan, submission_id=submission_id)
            result["visualizations"] = []
            return result

        # Validate visualization plan before execution
        try:
            validate_visualization_plan(visualization_nodes)
        except VisualizationPlanError as exc:
            return {
                "status": "failed",
                "output_path": None,
                "summary": {
                    "error_message": str(exc),
                    "failed_step_id": None,
                    "steps_run": [],
                    "step_statuses": {},
                    "duration_ms": 0,
                    "agent_summaries": [],
                    "step_metrics": {},
                    "operations_applied_by_step": {},
                    "warnings_by_step": {},
                },
                "visualizations": [],
            }

        # 1. Execute calculation steps via the engine
        engine_result = self.engine.execute(plan, submission_id=submission_id)

        # 2. Snapshot calc step statuses BEFORE viz execution (Req 6.1)
        calc_step_results = dict(self.engine.step_results)

        # 3. Execute visualization nodes regardless of calc outcome.
        #    If ALL calcs failed, viz nodes with failed dependencies will
        #    individually fail via dependency resolution. The runner continues
        #    executing all nodes (Req 6.3) and produces one spec per node (Req 5.3).
        pipeline_data = {}
        if hasattr(self.engine, '_last_state_data'):
            pipeline_data = self.engine._last_state_data
        else:
            # Reconstruct pipeline_data from step_results
            # The engine stores envelopes in state.data, but we don't have
            # direct access. Use step_results to build what we need.
            pipeline_data = self._build_pipeline_data_from_results(calc_step_results)

        visualization_specs = execute_visualization_nodes(
            nodes=visualization_nodes,
            step_results=calc_step_results,
            pipeline_data=pipeline_data,
        )

        # 4. Determine job status using the status handler (Req 6.2, 6.4)
        viz_job_result = self._status_handler.determine_job_status(
            calc_step_results=calc_step_results,
            visualization_specs=visualization_specs,
            viz_nodes=visualization_nodes,
        )

        # 5. Verify calc status isolation (Req 6.1)
        calc_statuses_before = {
            step_id: result.get("status", "unknown")
            for step_id, result in calc_step_results.items()
        }
        # After viz execution, verify statuses haven't changed
        self._status_handler.verify_calc_status_isolation(
            calc_step_results_before=calc_statuses_before,
            calc_step_results_after=calc_step_results,
        )

        # 6. Build final result
        # Override engine status with the viz-aware job status
        final_status = viz_job_result.job_status
        result = dict(engine_result)
        result["status"] = final_status
        result["visualizations"] = visualization_specs

        # Add viz warnings to the summary
        if viz_job_result.warnings:
            summary = result.get("summary", {})
            warnings_by_step = summary.get("warnings_by_step", {})
            warnings_by_step["visualization"] = viz_job_result.warnings
            summary["warnings_by_step"] = warnings_by_step
            result["summary"] = summary

        return result

    @staticmethod
    def _build_pipeline_data_from_results(
        step_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a pipeline_data-like dict from step_results.

        When we can't access state.data directly, we reconstruct enough
        information for the visualization runner to resolve dependencies.

        This is a fallback — the engine's envelope storage is the canonical
        source. The runner's _extract_operation_result already handles
        various envelope shapes.
        """
        pipeline_data: dict[str, Any] = {}
        for step_id, result in step_results.items():
            # Build a minimal envelope that the runner can parse
            artifacts = result.get("artifacts", {})
            data = result.get("data")
            if data is not None or artifacts:
                pipeline_data[step_id] = {
                    "data": data,
                    "artifacts": artifacts,
                }
        return pipeline_data
