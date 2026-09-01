import os
import sys
from unittest.mock import patch

import pandas as pd


from finflow_agent.agents.filter_agent import FilterAgent
from finflow_agent.operations.schemas import FilterCondition, FilterOperationPlan
from finflow_agent.planning.intent_package import PackageStatus
from finflow_agent.planning.package_builder import build_intent_package
from finflow_agent.tools.dataframe_profile import profile_dataframe
from finflow_agent.tools.predicate_grounder import (
    GroundingCandidate,
    LLMGroundingDecision,
    UnresolvedFilterClause,
    ground_filter_clauses,
)
from finflow_agent.tools.semantic_column_profiler import (
    BroadSemanticType,
    profile_semantic_columns,
)


def test_dataframe_profile_masks_sensitive_values_and_exposes_rich_samples():
    df = pd.DataFrame(
        {
            "Customer_Email": ["alice@example.com", "bob@example.com", "alice@example.com"],
            "Card_Number": ["4111111111111111", "5555555555554444", "4111111111111111"],
            "Product_Name": ["Headphones", "Coffee", "Tablet"],
        }
    )

    profile = profile_dataframe(df, include_samples=True, sample_rows=5)

    assert profile["row_count"] == 3
    assert profile["null_counts"]["Customer_Email"] == 0
    email_profile = next(col for col in profile.columns if col.column == "Customer_Email")
    card_profile = next(col for col in profile.columns if col.column == "Card_Number")

    assert "alice@example.com" not in repr(email_profile.representative_values)
    assert "4111111111111111" not in repr(card_profile.representative_values)
    assert email_profile.frequent_values
    assert card_profile.random_distinct_values
    assert profile["likely_date_columns"] == []


def test_predicate_grounder_routes_product_literal_away_from_payment_column():
    df = pd.DataFrame(
        {
            "Product_Name": ["Headphones", "Coffee", "Tablet", "Tab"],
            "Payment_Method": ["pay pal", "creditcard", "credit card", "PayPal"],
        }
    )

    profile = profile_dataframe(df, include_samples=True, sample_rows=5)
    semantic_profiles = profile_semantic_columns(profile)

    result = ground_filter_clauses(
        [
            UnresolvedFilterClause(
                requested_field="__merchant_column__",
                operator="eq",
                value="laptop electronics",
            )
        ],
        profile=profile,
        semantic_profiles=semantic_profiles,
    )

    assert result.status == "grounded"
    assert result.grounded_clauses[0].resolved_column == "Product_Name"
    assert all(clause.resolved_column != "Payment_Method" for clause in result.grounded_clauses)


def test_predicate_grounder_llm_fallback_is_independent_of_column_resolution_gate(monkeypatch):
    df = pd.DataFrame(
        {
            "alpha": ["A1", "A2", "A3", "A4"],
            "beta": ["x", "y", "z", "w"],
            "gamma": ["m", "n", "o", "p"],
            "delta": ["completed", "pending", "completed", "processing"],
        }
    )

    profile = profile_dataframe(df, include_samples=True, sample_rows=5)
    semantic_profiles = profile_semantic_columns(profile)

    monkeypatch.setenv("ENABLE_LLM_COLUMN_RESOLUTION", "false")
    monkeypatch.setenv("ENABLE_LLM_PREDICATE_GROUNDING", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    mocked_scores = {
        "alpha": GroundingCandidate(
            column="alpha",
            score=0.41,
            broad_type=BroadSemanticType.free_text,
            positive_evidence=["synthetic ambiguous candidate"],
            negative_evidence=[],
            semantic_tags=["alpha"],
        ),
        "beta": GroundingCandidate(
            column="beta",
            score=0.36,
            broad_type=BroadSemanticType.free_text,
            positive_evidence=["synthetic runner-up"],
            negative_evidence=[],
            semantic_tags=["beta"],
        ),
        "gamma": GroundingCandidate(
            column="gamma",
            score=0.11,
            broad_type=BroadSemanticType.free_text,
            positive_evidence=["synthetic low score"],
            negative_evidence=[],
            semantic_tags=["gamma"],
        ),
        "delta": GroundingCandidate(
            column="delta",
            score=0.09,
            broad_type=BroadSemanticType.free_text,
            positive_evidence=["synthetic low score"],
            negative_evidence=[],
            semantic_tags=["delta"],
        ),
    }

    with patch(
        "finflow_agent.tools.predicate_grounder._score_candidate",
        side_effect=lambda clause, semantic: mocked_scores[semantic.column],
    ), patch(
        "finflow_agent.tools.predicate_grounder._llm_ground_clause",
        return_value=LLMGroundingDecision(
            selected_column="beta",
            reason="explicit bounded fallback selection",
            confidence=0.91,
        ),
    ) as mock_llm_fallback:
        result = ground_filter_clauses(
            [
                UnresolvedFilterClause(
                    requested_field="activity",
                    operator="eq",
                    value="completed",
                )
            ],
            profile=profile,
            semantic_profiles=semantic_profiles,
        )

    mock_llm_fallback.assert_called_once()
    assert result.status == "grounded"
    assert result.grounded_clauses[0].resolved_column == "beta"
    assert result.grounded_clauses[0].grounding_method == "llm"


def test_predicate_grounder_invokes_llm_for_scores_below_shared_threshold(monkeypatch):
    df = pd.DataFrame(
        {
            "Product_Name": ["Headphones", "Coffee", "Tablet", "Camera"],
            "Misc_Notes": ["north", "south", "east", "west"],
        }
    )

    profile = profile_dataframe(df, include_samples=True, sample_rows=5)
    semantic_profiles = profile_semantic_columns(profile)

    monkeypatch.setenv("ENABLE_LLM_PREDICATE_GROUNDING", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    mocked_scores = {
        "Product_Name": GroundingCandidate(
            column="Product_Name",
            score=0.72,
            broad_type=BroadSemanticType.product,
            positive_evidence=["best score below shared threshold"],
            negative_evidence=[],
            semantic_tags=["product"],
        ),
        "Misc_Notes": GroundingCandidate(
            column="Misc_Notes",
            score=0.40,
            broad_type=BroadSemanticType.free_text,
            positive_evidence=["weaker backup candidate"],
            negative_evidence=[],
            semantic_tags=["notes"],
        ),
    }

    with patch(
        "finflow_agent.tools.predicate_grounder._score_candidate",
        side_effect=lambda clause, semantic: mocked_scores[semantic.column],
    ), patch(
        "finflow_agent.tools.predicate_grounder._llm_ground_clause",
        return_value=LLMGroundingDecision(
            selected_column="Product_Name",
            reason="product-like free text",
            confidence=0.88,
        ),
    ) as mock_llm_fallback:
        result = ground_filter_clauses(
            [
                UnresolvedFilterClause(
                    requested_field="catalog",
                    operator="eq",
                    value="laptop",
                )
            ],
            profile=profile,
            semantic_profiles=semantic_profiles,
        )

    mock_llm_fallback.assert_called_once()
    assert result.status == "grounded"
    assert result.grounded_clauses[0].resolved_column == "Product_Name"
    assert result.grounded_clauses[0].grounding_method == "llm"


def test_filter_agent_returns_empty_result_when_value_is_absent():
    df = pd.DataFrame(
        {
            "Product_Name": ["Headphones", "Coffee", "Tablet"],
            "Payment_Method": ["PayPal", "Card", "Cash"],
        }
    )
    plan = FilterOperationPlan(
        conditions=[
            FilterCondition(
                column="__merchant_column__",
                operator="eq",
                value="laptop electronics",
            )
        ],
        select_columns=["Product_Name"],
    )

    profile = profile_dataframe(df, include_samples=True, sample_rows=5)
    intent_package = build_intent_package(
        submission_id="sub-zero-row",
        filter_plan=plan,
        profile=profile,
    )

    assert intent_package.status == PackageStatus.VALID

    result = FilterAgent().execute(
        {"plan": plan.model_dump()},
        {"input_dataframe": df, "intent_package": intent_package},
    )

    assert result.status == "success"
    assert result.error_message is None
    assert result.data is not None
    assert result.data.empty

    value_resolution = result.artifacts.get("value_resolution") or []
    assert value_resolution
    assert value_resolution[0]["matched"] is False
    assert value_resolution[0]["requested_value"] == "laptop electronics"
    assert result.warnings
    assert "zero rows" in " ".join(result.warnings).lower()
    assert intent_package.status == PackageStatus.VALID
