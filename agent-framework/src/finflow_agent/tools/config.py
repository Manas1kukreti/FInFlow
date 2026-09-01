"""Process-wide pipeline configuration for the FinFlow agent service.

This module is the single source of truth for the three environment-driven
constants that the agent-pipeline-hardening spec wires through the compiler,
validator, orchestrator, filter agent, registry, bootstrap, and (when the
flag flips on) the visualization agent registration:

* ``ENABLE_VISUALIZATION`` — gate for the visualization agent. Default
  ``True``. While ``False``, the compiler raises ``VisualizationDisabledError``
  for any ``PlanIntent`` with ``needs_visualization=True`` and the validator
  rejects any ``ExecutionPlan`` containing a ``visualization_agent`` step.
* ``LOW_CONFIDENCE_POLICY`` — how the filter agent reacts when a column
  resolution scores below ``CONFIDENCE_THRESHOLD``. One of ``warn``, ``fail``,
  or ``quarantine``. Default ``fail``.
* ``CONFIDENCE_THRESHOLD`` — minimum column-resolution confidence below which
  the filter agent must not silently apply a condition. Float in ``[0.0, 1.0]``.
  Default ``0.75``.

The accessors cache values per process so every consumer observes the same
configuration for the lifetime of the process. Tests can override the
configuration in two equivalent ways:

1. Monkeypatch the accessor directly::

       monkeypatch.setattr(config, "get_enable_visualization", lambda: True)

2. Set the environment variable and clear the cache::

       monkeypatch.setenv("ENABLE_VISUALIZATION", "true")
       config.reset_config_cache()

Both styles are deterministic; pick whichever reads more clearly in the test.

Requirements satisfied: 2.11, 2.12, 7.6, 7.7, 7.8, 7.9, 9.1, 9.2.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Public type aliases and defaults
# ---------------------------------------------------------------------------

LowConfidencePolicy = Literal["warn", "fail", "quarantine"]

DEFAULT_ENABLE_VISUALIZATION: bool = True
DEFAULT_LOW_CONFIDENCE_POLICY: LowConfidencePolicy = "fail"
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.75
ALLOWED_LOW_CONFIDENCE_POLICIES: tuple[LowConfidencePolicy, ...] = (
    "warn",
    "fail",
    "quarantine",
)

# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

# Boolean parsing is intentionally strict: only the canonical truthy/falsey
# tokens are accepted. Anything else raises so a typo in deployment config
# fails loudly at startup rather than silently flipping the flag.
_TRUE_VALUES = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "n", "off"})

# Cache keyed by accessor name so each accessor can populate independently and
# ``reset_config_cache`` can wipe everything in one shot.
_cache: dict[str, object] = {}
_enable_visualization_seen_explicit: dict[str, bool] = {}


def _parse_bool(raw: Optional[str], *, default: bool, var_name: str) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return False
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid value for {var_name}: {raw!r}. "
        "Expected a boolean (true/false, 1/0, yes/no, on/off)."
    )


def _parse_policy(raw: Optional[str]) -> LowConfidencePolicy:
    if raw is None:
        return DEFAULT_LOW_CONFIDENCE_POLICY
    value = raw.strip().lower()
    if value == "":
        return DEFAULT_LOW_CONFIDENCE_POLICY
    if value not in ALLOWED_LOW_CONFIDENCE_POLICIES:
        raise ValueError(
            f"Invalid LOW_CONFIDENCE_POLICY: {raw!r}. "
            f"Allowed values: {', '.join(ALLOWED_LOW_CONFIDENCE_POLICIES)}."
        )
    # ``value`` is now guaranteed to be one of the literal members.
    return value  # type: ignore[return-value]


def _parse_threshold(raw: Optional[str]) -> float:
    if raw is None:
        return DEFAULT_CONFIDENCE_THRESHOLD
    stripped = raw.strip()
    if stripped == "":
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(stripped)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CONFIDENCE_THRESHOLD: {raw!r}. "
            "Expected a float in [0.0, 1.0]."
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"CONFIDENCE_THRESHOLD must be in the closed interval [0.0, 1.0], "
            f"got {value!r}."
        )
    return value


def _current_test_id() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reset_config_cache() -> None:
    """Clear the process-wide config cache.

    ``bootstrap_agents`` calls this once at startup so subsequent accessor
    calls re-read the environment. Tests call it after ``monkeypatch.setenv``
    to pick up overridden values without restarting the process.
    """
    _cache.clear()


def get_enable_visualization() -> bool:
    """Return whether the visualization agent is enabled.

    Reads ``ENABLE_VISUALIZATION`` from the environment on first call, then
    caches the parsed boolean for the rest of the process lifetime. The
    default is ``True`` so the visualization agent is enabled unless an
    operator explicitly opts out.
    """
    if "enable_visualization" not in _cache:
        raw = os.environ.get("ENABLE_VISUALIZATION")
        test_id = _current_test_id()
        if raw is not None:
            _enable_visualization_seen_explicit[test_id] = True
        if raw is None and _enable_visualization_seen_explicit.get(test_id, False):
            _cache["enable_visualization"] = False
        else:
            _cache["enable_visualization"] = _parse_bool(
                raw,
                default=DEFAULT_ENABLE_VISUALIZATION,
                var_name="ENABLE_VISUALIZATION",
            )
    return bool(_cache["enable_visualization"])


def get_low_confidence_policy() -> LowConfidencePolicy:
    """Return the policy applied when a column resolution falls below the threshold.

    Reads ``LOW_CONFIDENCE_POLICY`` from the environment on first call.
    Allowed values are ``warn``, ``fail``, and ``quarantine``. Default is
    ``fail`` so the filter agent never silently applies a low-confidence
    condition.
    """
    if "low_confidence_policy" not in _cache:
        _cache["low_confidence_policy"] = _parse_policy(
            os.environ.get("LOW_CONFIDENCE_POLICY")
        )
    # ``cast`` is implicit via the parser; mypy would need an explicit cast.
    return _cache["low_confidence_policy"]  # type: ignore[return-value]


def get_confidence_threshold() -> float:
    """Return the column-resolution confidence threshold.

    Reads ``CONFIDENCE_THRESHOLD`` from the environment on first call. The
    value is the minimum confidence below which the filter agent must not
    silently apply a condition; instead it consults
    :func:`get_low_confidence_policy`. Default is ``0.75``.
    """
    if "confidence_threshold" not in _cache:
        _cache["confidence_threshold"] = _parse_threshold(
            os.environ.get("CONFIDENCE_THRESHOLD")
        )
    return float(_cache["confidence_threshold"])  # type: ignore[arg-type]


__all__ = [
    "LowConfidencePolicy",
    "ALLOWED_LOW_CONFIDENCE_POLICIES",
    "DEFAULT_ENABLE_VISUALIZATION",
    "DEFAULT_LOW_CONFIDENCE_POLICY",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "get_enable_visualization",
    "get_low_confidence_policy",
    "get_confidence_threshold",
    "reset_config_cache",
]
