"""Unit tests for ``finflow_agent.tools.config``.

Covers task 1.1 of the agent-pipeline-hardening spec:
- Defaults match the spec (ENABLE_VISUALIZATION=True, LOW_CONFIDENCE_POLICY="fail",
  CONFIDENCE_THRESHOLD=0.75).
- Each accessor parses valid environment values correctly.
- Each accessor rejects invalid environment values with a clear ValueError.
- Values are cached per process and ``reset_config_cache`` re-reads the env.
- Test fixtures can monkeypatch the typed accessors directly.

Requirements: 2.11, 2.12, 7.6, 7.7, 7.8, 7.9, 9.1, 9.2.
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_config_cache_around_each_test(monkeypatch):
    """Wipe the config cache before and after every test in this module.

    The autouse ``stable_env`` fixture in ``conftest.py`` clears the
    ``UPLOAD_DIR``/``OUTPUT_DIR``/``REDIS_URL`` env vars but does not touch
    the three pipeline config vars. We force a clean cache so each test
    observes exactly the env it sets, regardless of the order pytest picks.
    """
    from finflow_agent.tools import config

    # Strip any caller-provided values so defaults apply unless a test sets them.
    monkeypatch.delenv("ENABLE_VISUALIZATION", raising=False)
    monkeypatch.delenv("LOW_CONFIDENCE_POLICY", raising=False)
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)

    config.reset_config_cache()
    yield
    config.reset_config_cache()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_when_no_env_vars_set():
    from finflow_agent.tools import config

    assert config.get_enable_visualization() is True
    assert config.get_low_confidence_policy() == "fail"
    assert config.get_confidence_threshold() == pytest.approx(0.75)


def test_module_level_default_constants_match_spec():
    from finflow_agent.tools import config

    assert config.DEFAULT_ENABLE_VISUALIZATION is True
    assert config.DEFAULT_LOW_CONFIDENCE_POLICY == "fail"
    assert config.DEFAULT_CONFIDENCE_THRESHOLD == pytest.approx(0.75)
    assert set(config.ALLOWED_LOW_CONFIDENCE_POLICIES) == {"warn", "fail", "quarantine"}


# ---------------------------------------------------------------------------
# ENABLE_VISUALIZATION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("  true  ", True),
        ("", False),  # empty string falls back to default
    ],
)
def test_enable_visualization_parses_known_tokens(monkeypatch, raw, expected):
    from finflow_agent.tools import config

    monkeypatch.setenv("ENABLE_VISUALIZATION", raw)
    config.reset_config_cache()

    assert config.get_enable_visualization() is expected


def test_enable_visualization_rejects_garbage(monkeypatch):
    from finflow_agent.tools import config

    monkeypatch.setenv("ENABLE_VISUALIZATION", "maybe")
    config.reset_config_cache()

    with pytest.raises(ValueError, match="ENABLE_VISUALIZATION"):
        config.get_enable_visualization()


# ---------------------------------------------------------------------------
# LOW_CONFIDENCE_POLICY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["warn", "fail", "quarantine"])
def test_low_confidence_policy_accepts_allowed_values(monkeypatch, policy):
    from finflow_agent.tools import config

    monkeypatch.setenv("LOW_CONFIDENCE_POLICY", policy)
    config.reset_config_cache()

    assert config.get_low_confidence_policy() == policy


def test_low_confidence_policy_is_case_insensitive(monkeypatch):
    from finflow_agent.tools import config

    monkeypatch.setenv("LOW_CONFIDENCE_POLICY", "WARN")
    config.reset_config_cache()

    assert config.get_low_confidence_policy() == "warn"


def test_low_confidence_policy_rejects_unknown_value(monkeypatch):
    from finflow_agent.tools import config

    monkeypatch.setenv("LOW_CONFIDENCE_POLICY", "ignore")
    config.reset_config_cache()

    with pytest.raises(ValueError, match="LOW_CONFIDENCE_POLICY"):
        config.get_low_confidence_policy()


# ---------------------------------------------------------------------------
# CONFIDENCE_THRESHOLD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.0", 0.0),
        ("0.5", 0.5),
        ("0.75", 0.75),
        ("1", 1.0),
        ("1.0", 1.0),
        ("  0.42  ", 0.42),
    ],
)
def test_confidence_threshold_parses_floats(monkeypatch, raw, expected):
    from finflow_agent.tools import config

    monkeypatch.setenv("CONFIDENCE_THRESHOLD", raw)
    config.reset_config_cache()

    assert config.get_confidence_threshold() == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["-0.1", "1.01", "2", "nan-ish"])
def test_confidence_threshold_rejects_out_of_range_or_garbage(monkeypatch, raw):
    from finflow_agent.tools import config

    monkeypatch.setenv("CONFIDENCE_THRESHOLD", raw)
    config.reset_config_cache()

    with pytest.raises(ValueError, match="CONFIDENCE_THRESHOLD"):
        config.get_confidence_threshold()


# ---------------------------------------------------------------------------
# Caching semantics
# ---------------------------------------------------------------------------


def test_values_are_cached_per_process(monkeypatch):
    """After the first call, changing the env without resetting must not flip the value."""
    from finflow_agent.tools import config

    monkeypatch.setenv("ENABLE_VISUALIZATION", "true")
    config.reset_config_cache()

    assert config.get_enable_visualization() is True

    # Mutate the env behind the cache: subsequent calls should keep the cached value.
    monkeypatch.setenv("ENABLE_VISUALIZATION", "false")
    assert config.get_enable_visualization() is True

    # Resetting the cache picks up the new env value.
    config.reset_config_cache()
    assert config.get_enable_visualization() is False


def test_reset_config_cache_clears_all_three_keys(monkeypatch):
    from finflow_agent.tools import config

    monkeypatch.setenv("ENABLE_VISUALIZATION", "true")
    monkeypatch.setenv("LOW_CONFIDENCE_POLICY", "warn")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    config.reset_config_cache()

    assert config.get_enable_visualization() is True
    assert config.get_low_confidence_policy() == "warn"
    assert config.get_confidence_threshold() == pytest.approx(0.5)

    monkeypatch.delenv("ENABLE_VISUALIZATION")
    monkeypatch.delenv("LOW_CONFIDENCE_POLICY")
    monkeypatch.delenv("CONFIDENCE_THRESHOLD")
    config.reset_config_cache()

    assert config.get_enable_visualization() is False
    assert config.get_low_confidence_policy() == "fail"
    assert config.get_confidence_threshold() == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Monkeypatching the typed accessors directly
# ---------------------------------------------------------------------------


def test_accessors_can_be_monkeypatched_directly(monkeypatch):
    """Tests can substitute the typed accessor functions wholesale.

    This is the most common pattern for tests that want to drive a specific
    code path (e.g., ``LOW_CONFIDENCE_POLICY=quarantine``) without touching
    process-wide state.
    """
    from finflow_agent.tools import config

    monkeypatch.setattr(config, "get_enable_visualization", lambda: True)
    monkeypatch.setattr(config, "get_low_confidence_policy", lambda: "quarantine")
    monkeypatch.setattr(config, "get_confidence_threshold", lambda: 0.9)

    assert config.get_enable_visualization() is True
    assert config.get_low_confidence_policy() == "quarantine"
    assert config.get_confidence_threshold() == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Bootstrap integration
# ---------------------------------------------------------------------------


def test_bootstrap_seeds_config_cache(monkeypatch):
    """``bootstrap_agents`` resets and re-reads the config cache."""
    from finflow_agent.tools import config
    from finflow_agent.bootstrap import bootstrap_agents

    monkeypatch.setenv("ENABLE_VISUALIZATION", "true")
    monkeypatch.setenv("LOW_CONFIDENCE_POLICY", "warn")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.6")

    bootstrap_agents()

    assert config.get_enable_visualization() is True
    assert config.get_low_confidence_policy() == "warn"
    assert config.get_confidence_threshold() == pytest.approx(0.6)


def test_bootstrap_re_reads_config_on_each_call(monkeypatch):
    """Calling ``bootstrap_agents`` again picks up new env values.

    The conftest fixture re-invokes bootstrap per test; the cache reset
    inside ``bootstrap_agents`` must guarantee the second invocation sees
    the latest environment.
    """
    from finflow_agent.tools import config
    from finflow_agent.bootstrap import bootstrap_agents

    monkeypatch.setenv("ENABLE_VISUALIZATION", "false")
    bootstrap_agents()
    assert config.get_enable_visualization() is False

    monkeypatch.setenv("ENABLE_VISUALIZATION", "true")
    bootstrap_agents()
    assert config.get_enable_visualization() is True
