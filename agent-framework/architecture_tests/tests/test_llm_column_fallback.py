"""Tests for the constrained LLM column resolution fallback (Tier 5).

Verifies that when deterministic tiers (exact, normalized, synonym, fuzzy)
all produce scores below CONFIDENCE_THRESHOLD, the resolver attempts a
constrained LLM call that can ONLY select from the actual available columns.

The LLM is always mocked — no real API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finflow_agent.tools.column_resolver import (
    CONFIDENCE_THRESHOLD,
    ColumnResolution,
    resolve_column,
    _LLM_RESOLUTION_CONFIDENCE,
    _LLMColumnChoice,
)
from finflow_agent.tools.dataframe_profile import profile_dataframe
from finflow_agent.planning.package_builder import build_intent_package
from finflow_agent.tools.predicate_grounder import LLMGroundingDecision


_LLM_PATCH_TARGET = "finflow_agent.llm.get_chat_groq"


@pytest.fixture
def sample_profile():
    """Profile with columns that stay below the deterministic LLM threshold."""
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Reference": ["REF001"],
        "Payment Mode": ["UPI"],
        "Amount": [1500.00],
        "Quantity": [1],
    })
    return profile_dataframe(df, include_samples=False)


@pytest.fixture(autouse=True)
def enable_llm_resolution(monkeypatch):
    """Enable LLM resolution and provide a fake API key for all tests."""
    monkeypatch.setenv("ENABLE_LLM_COLUMN_RESOLUTION", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    yield


def _mock_llm_returning(selected_column, reason="test"):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _LLMColumnChoice(
        selected_column=selected_column, reason=reason,
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# 1. LLM picks the right column when fuzzy fails
# ---------------------------------------------------------------------------

def test_llm_fallback_resolves_settlement_gateway_to_payment_mode(sample_profile):
    mock_llm = _mock_llm_returning(
        "Payment Mode",
        "settlement gateway = payment method",
    )
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__settlement_gateway_column__", sample_profile)
    assert result.confidence == _LLM_RESOLUTION_CONFIDENCE
    assert result.matched_column == "Payment Mode"
    assert "llm_semantic_match" in result.reason


def test_llm_fallback_resolves_settlement_gateway_to_payment_mode_again(sample_profile):
    mock_llm = _mock_llm_returning(
        "Payment Mode",
        "settlement gateway = payment channel",
    )
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("settlement_gateway", sample_profile)
    assert result.confidence == _LLM_RESOLUTION_CONFIDENCE
    assert result.matched_column == "Payment Mode"


# ---------------------------------------------------------------------------
# 2. LLM cannot invent columns
# ---------------------------------------------------------------------------

def test_llm_fallback_rejects_invented_column(sample_profile):
    mock_llm = _mock_llm_returning("Merchant Name", "invented")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__mystery_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD
    assert "llm_semantic_match" not in result.reason


# ---------------------------------------------------------------------------
# 3. LLM returns null
# ---------------------------------------------------------------------------

def test_llm_fallback_returns_null_falls_through(sample_profile):
    mock_llm = _mock_llm_returning(None, "no match")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__mystery_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 4. Graceful failure
# ---------------------------------------------------------------------------

def test_llm_fallback_graceful_on_network_error(sample_profile):
    with patch(_LLM_PATCH_TARGET, side_effect=Exception("timeout")):
        result = resolve_column("__mystery_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


def test_llm_fallback_graceful_on_parse_error(sample_profile):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = "garbage"
    mock_llm.with_structured_output.return_value = mock_structured
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__mystery_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 5. Disabled via env
# ---------------------------------------------------------------------------

def test_llm_fallback_disabled_via_env(sample_profile, monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_COLUMN_RESOLUTION", "false")
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("__mystery_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 6. LLM not called when deterministic tiers succeed
# ---------------------------------------------------------------------------

def test_llm_not_called_for_exact_match(sample_profile):
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("Amount", sample_profile)
    assert result.confidence == 1.0
    assert result.matched_column == "Amount"


def test_llm_not_called_for_fuzzy_above_threshold(monkeypatch):
    df = pd.DataFrame({"Customer_Age": [25, 30], "Name": ["A", "B"]})
    profile = profile_dataframe(df, include_samples=False)
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("customer_age", profile)
    assert result.confidence >= CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 7. End-to-end filter agent integration
# ---------------------------------------------------------------------------

def test_filter_agent_succeeds_with_llm_resolved_column(bootstrap_agents):
    from finflow_agent.agents.filter_agent import FilterAgent

    df = pd.DataFrame({
        "alpha": ["X", "Y", "X"],
        "beta": ["A", "B", "A"],
        "gamma": ["M", "N", "M"],
        "delta": ["Q", "R", "Q"],
    })
    plan = {
        "conditions": [
            {
                "column": "__mystery_column__",
                "operator": "eq",
                "value": "B",
            }
        ],
        "logic": "and",
    }
    with patch(
        "finflow_agent.tools.predicate_grounder._llm_ground_clause",
        return_value=LLMGroundingDecision(
            selected_column="beta",
            reason="mystery field resolves to beta",
            confidence=0.91,
        ),
    ):
        result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    assert result.status == "success", result.error_message
    assert len(result.data) == 1
    assert result.data.iloc[0]["beta"] == "B"
    mapping = result.artifacts.get("column_mapping", [])
    assert len(mapping) == 1
    assert mapping[0]["matched_column"] == "beta"
    assert mapping[0]["confidence"] >= 0.91
    assert "semantic grounding (llm)" in mapping[0]["reason"]


def test_unresolved_semantic_field_can_still_use_llm_fallback():
    from finflow_agent.operations.schemas import FilterOperationPlan
    from finflow_agent.tools.dataframe_profile import profile_dataframe

    profile = profile_dataframe(
        pd.DataFrame(
            {
                "education_level": ["Master"],
                "reference_code": ["REF001"],
                "amount": [1500.00],
            }
        ),
        include_samples=False,
    )

    plan = FilterOperationPlan.model_validate(
        {
            "conditions": [
                {
                    "column": "qualification",
                    "operator": "eq",
                    "value": "abc",
                }
            ],
            "logic": "and",
        }
    )

    with patch(
        "finflow_agent.tools.predicate_grounder._llm_ground_clause",
        return_value=LLMGroundingDecision(
            selected_column="education_level",
            reason="qualification maps to education_level",
            confidence=0.88,
        ),
    ) as mock_llm:
        pkg = build_intent_package(
            submission_id="llm-fallback",
            filter_plan=plan,
            profile=profile,
        )

    assert mock_llm.call_count == 1
    assert pkg.status.value in {"valid", "needs_review"}
    assert pkg.resolved_columns
    assert pkg.resolved_columns[0].resolved_column == "education_level"
