"""Plan validator for the FinFlow Agent Service.

Performs every check from design Component 3 and the ``validate_plan``
algorithm of the agent-pipeline-hardening spec, in this order:

    1. Unique ``step_id`` values.
    2. Every ``step.agent`` is registered.
    3. Disabled future agents (currently only ``visualization_agent`` when
       ``ENABLE_VISUALIZATION=false``) cannot run; the canonical message is
       ``"visualization_agent is not enabled in this version"``.
    4. Every ``depends_on`` entry references an existing ``step_id``.
    5. The dependency graph is acyclic (Kahn's topological sort completes).
    6. Every ``input_from`` key was produced by a strictly-earlier step in
       topological order. Format:
       ``"Step <id> input_from '<key>' not produced earlier"``.
    7. Stage ordering is monotonic across
       ``ingest → transform → analyze → visualize → deliver``.
    8. Per-step ``params`` re-validate against the agent's registered
       Pydantic param model. On failure the Pydantic error and the offending
       ``step_id`` are surfaced.

On success returns ``(True, "")`` without mutating the plan.

Requirements satisfied: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 9.1,
11.2, 11.3.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

from pydantic import ValidationError

from finflow_agent.agents.visualization_agent import VISUALIZATION_DISABLED_MESSAGE
from finflow_agent.registry import AGENT_PARAM_MODELS, registry
from finflow_agent.state import ExecutionPlan


# Stage ordering used by the monotonic-progression check. A step may only
# depend on steps with the same or lower rank (= strictly-earlier-in-pipeline).
# A ``dep_rank > my_rank`` violates the ``ingest → transform → analyze →
# visualize → deliver`` flow and is rejected.
_STAGE_RANK: Dict[str, int] = {
    "ingest": 1,
    "transform": 2,
    "analyze": 3,
    "visualize": 4,
    "deliver": 5,
}


def validate_plan(plan: ExecutionPlan) -> Tuple[bool, str]:
    """Validate an ``ExecutionPlan`` before the engine touches it.

    Returns ``(True, "")`` only when every check passes. On any failure,
    returns ``(False, error_message)`` whose message names the offending
    step and the failing rule. The plan is never mutated.
    """
    steps = plan.steps

    # 1. Unique step_ids
    seen_ids: Set[str] = set()
    for step in steps:
        if step.step_id in seen_ids:
            return False, f"Duplicate step_id: {step.step_id}"
        seen_ids.add(step.step_id)

    # 2. Every agent is registered, and disabled future agents cannot run.
    #    The disabled check is intentionally co-located with the registry
    #    lookup so the canonical "Unknown agent: <name>" error never fires
    #    for the legitimately-named ``visualization_agent`` slot (bootstrap
    #    keeps it registered with ``enabled=False`` whenever the
    #    ``ENABLE_VISUALIZATION`` flag is off).
    for step in steps:
        try:
            spec = registry.get_spec(step.agent)
        except ValueError:
            return False, f"Unknown agent: {step.agent}"

        if not spec.enabled:
            # Today only ``visualization_agent`` is ever disabled; surface
            # the canonical message defined alongside that agent so the
            # orchestrator (task 12.1) can recognize it for quarantine
            # routing. The generic branch is defense in depth for any
            # future disabled-by-default agent.
            if step.agent == "visualization_agent":
                return False, VISUALIZATION_DISABLED_MESSAGE
            return False, (
                f"Agent '{step.agent}' is registered but disabled in this version"
            )

    # 3. depends_on entries reference existing step_ids
    step_map = {step.step_id: step for step in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep not in step_map:
                return False, (
                    f"Step '{step.step_id}' depends on unknown step '{dep}'"
                )

    # 4. Topological sort (cycle detection) via Kahn's algorithm. A separate
    #    in-degree copy is used so we never mutate any state derived from
    #    ``plan``; the validator must be side-effect free on success and
    #    failure alike.
    adj: Dict[str, List[str]] = {step.step_id: [] for step in steps}
    in_degree: Dict[str, int] = {step.step_id: 0 for step in steps}
    for step in steps:
        for dep in step.depends_on:
            adj[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue: deque = deque(sid for sid, deg in in_degree.items() if deg == 0)
    sorted_ids: List[str] = []
    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_ids) != len(steps):
        return False, "Cycle detected in ExecutionPlan"

    # 5. input_from keys must be produced by a strictly-earlier step in
    #    topological order. The produced-key for each step is its
    #    ``output_key`` when set, otherwise its ``step_id`` (matching what
    #    the engine stores in ``state.data``).
    produced_keys: Set[str] = set()
    for sid in sorted_ids:
        step = step_map[sid]
        for inp in step.input_from:
            if inp not in produced_keys:
                return False, (
                    f"Step {sid} input_from '{inp}' not produced earlier"
                )
        produced_keys.add(step.output_key or step.step_id)

    # 6. Monotonic stage ordering across ingest → transform → analyze →
    #    visualize → deliver. A step may only depend on steps at the same
    #    or earlier stage; a higher-ranked dependency violates the flow.
    for step in steps:
        my_rank = _STAGE_RANK.get(registry.get_spec(step.agent).stage, 0)
        for dep_id in step.depends_on:
            dep_step = step_map[dep_id]
            dep_rank = _STAGE_RANK.get(
                registry.get_spec(dep_step.agent).stage, 0
            )
            if dep_rank > my_rank:
                return False, (
                    f"Stage ordering violation at {step.step_id}: "
                    f"depends on '{dep_id}' which is at a later stage"
                )

    # 7. Per-step params re-validate against the registered Pydantic model.
    #    Steps whose agents declare a ``params_model`` (ingestion, cleaning,
    #    filter, reporting, visualization) are gated here so malformed params
    #    cannot reach the engine. Agents without a registered model
    #    (currently only ``calculation_agent``) are intentionally skipped
    #    here; the engine and the agent's own internal validation remain the
    #    next safety layers for them.
    for step in steps:
        param_model = AGENT_PARAM_MODELS.get(step.agent)
        if param_model is None:
            continue
        try:
            param_model.model_validate(step.params)
        except ValidationError as exc:
            return False, f"Invalid params for {step.step_id}: {exc}"

    return True, ""
