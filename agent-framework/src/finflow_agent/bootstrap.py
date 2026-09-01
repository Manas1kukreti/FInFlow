import logging

from finflow_agent.registry import registry
from finflow_agent.tools.config import (
    get_confidence_threshold,
    get_enable_visualization,
    get_low_confidence_policy,
    reset_config_cache,
)

logger = logging.getLogger(__name__)


def bootstrap_agents() -> None:
    """Bootstrap the FinFlow agent registry and seed pipeline configuration.

    Two responsibilities, intentionally co-located so a single startup call
    yields a fully consistent process:

    1. Read the three pipeline configuration constants exactly once at
       startup (``ENABLE_VISUALIZATION``, ``LOW_CONFIDENCE_POLICY``,
       ``CONFIDENCE_THRESHOLD``) via the typed accessors in
       :mod:`finflow_agent.tools.config`. This warms the per-process cache so
       the compiler, validator, orchestrator, filter agent, registry, and
       visualization agent registration all observe the same values for the
       lifetime of the process.

    2. Import every agent module so their ``@registry.register`` decorators
       populate the registry, then synchronize each agent's ``enabled`` flag
       with its corresponding configuration constant. The visualization agent
       is imported here so the validator's ``"Unknown agent:
       visualization_agent"`` error never fires; its ``AgentSpec.enabled`` is
       then flipped to ``False`` whenever ``ENABLE_VISUALIZATION`` is false,
       so the validator and compiler can reject any plan referencing the
       disabled slot with the canonical
       ``"visualization_agent is not enabled in this version"`` message
       (task 13.1).

    Requirements satisfied: 2.11, 2.12, 7.6, 7.7, 7.8, 7.9, 9.1, 9.2.
    """
    logger.info("Bootstrapping all agents...")

    # Seed the pipeline configuration cache. Resetting first ensures a clean
    # read in deployments that re-invoke ``bootstrap_agents`` (for example,
    # the test fixture in ``finflow_architecture_tests/tests/conftest.py``).
    reset_config_cache()
    enable_visualization = get_enable_visualization()
    low_confidence_policy = get_low_confidence_policy()
    confidence_threshold = get_confidence_threshold()
    logger.info(
        "Pipeline config: ENABLE_VISUALIZATION=%s, LOW_CONFIDENCE_POLICY=%s, "
        "CONFIDENCE_THRESHOLD=%s",
        enable_visualization,
        low_confidence_policy,
        confidence_threshold,
    )

    import finflow_agent.agents.ingestion_agent  # noqa: F401
    import finflow_agent.agents.cleaning_agent  # noqa: F401
    import finflow_agent.agents.filter_agent  # noqa: F401
    import finflow_agent.agents.calculation_agent  # noqa: F401
    import finflow_agent.agents.visualization_agent  # noqa: F401
    import finflow_agent.agents.reporting_agent  # noqa: F401

    # Flip the visualization agent's ``enabled`` flag to match
    # ``ENABLE_VISUALIZATION``. The agent is registered unconditionally above
    # so the plan validator never returns ``"Unknown agent: visualization_agent"``
    # for plans that legitimately name the slot. When the env flag is false
    # (the default in this version), we mark the spec disabled here so the
    # validator (task 8.1) and compiler (task 7.1) can reject any
    # ``visualization_agent`` step with the canonical
    # ``"visualization_agent is not enabled in this version"`` message. The
    # agent class's ``execute`` already returns a controlled failed
    # ``AgentResult`` with the same message regardless of this flag, so the
    # plumbing is defense in depth: validator/compiler refuse the plan up
    # front, and the engine path stays safe even if a plan ever slipped
    # through. We never touch any other agent's ``enabled`` flag.
    visualization_spec = registry.get_spec("visualization_agent")
    visualization_spec.enabled = bool(enable_visualization)
    logger.info(
        "Visualization agent registered with enabled=%s",
        visualization_spec.enabled,
    )

    logger.info("Agent bootstrapping complete.")


def validate_required_agents_registered() -> None:
    """
    Verify that all required agents are correctly registered.
    Fails loudly with a ValueError if any agent is missing.
    """
    required_agents = [
        "ingestion_agent",
        "cleaning_agent",
        "filter_agent",
        "calculation_agent",
        "reporting_agent",
    ]
    for name in required_agents:
        try:
            registry.get_spec(name)
        except ValueError as e:
            raise ValueError(f"Startup check failed: Required agent '{name}' is not registered.") from e
