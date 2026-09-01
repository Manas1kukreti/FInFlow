"""Tests for the post-LLM intent normalizer."""
import pytest
from finflow_agent.planning.intent_schema import PlanIntent
from finflow_agent.planning.intent_normalizer import normalize_intent


def _base_intent(**kwargs):
    from finflow_agent.operations.schemas import (
        CleaningOperationPlan, TrimWhitespaceOperation
    )
    defaults = {
        "needs_cleaning": True,
        "output_format": "xlsx",
        "cleaning_plan": CleaningOperationPlan(
            operations=[TrimWhitespaceOperation(columns="__all_string_columns__")]
        ),
    }
    defaults.update(kwargs)
    return PlanIntent(**defaults)


def test_normalizer_adds_select_columns_for_only_return():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "Clean this data and only return Customer Id",
        available_columns=["Customer_ID", "Product_Name", "Amount"],
    )
    assert result.needs_filtering is True
    assert result.filter_plan is not None
    assert result.filter_plan.select_columns is not None
    assert "Customer_ID" in result.filter_plan.select_columns


def test_normalizer_adds_select_columns_for_show_only():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "show only Product Name",
        available_columns=["Customer_ID", "Product_Name", "Amount"],
    )
    assert result.needs_filtering is True
    assert result.filter_plan.select_columns == ["Product_Name"]


def test_normalizer_handles_multiple_columns():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "only return Customer Id and Amount",
        available_columns=["Customer_ID", "Product_Name", "Amount"],
    )
    assert result.needs_filtering is True
    assert "Customer_ID" in result.filter_plan.select_columns
    assert "Amount" in result.filter_plan.select_columns


def test_normalizer_does_not_modify_when_select_columns_already_set():
    from finflow_agent.operations.schemas import FilterOperationPlan
    plan = FilterOperationPlan(
        conditions=[], select_columns=["Amount"], logic="and"
    )
    intent = _base_intent(needs_filtering=True, filter_plan=plan)
    result = normalize_intent(
        intent,
        "only return Customer Id",
        available_columns=["Customer_ID", "Amount"],
    )
    # Should NOT override the existing select_columns
    assert result.filter_plan.select_columns == ["Amount"]


def test_normalizer_does_nothing_for_normal_instructions():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "Clean this data and remove duplicates",
        available_columns=["Customer_ID", "Amount"],
    )
    assert result.needs_filtering is False
    assert result.filter_plan is None


def test_normalizer_works_without_available_columns():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "only return Customer Id",
        available_columns=None,
    )
    assert result.needs_filtering is True
    assert result.filter_plan.select_columns == ["Customer Id"]


def test_normalizer_handles_just_show():
    intent = _base_intent(needs_filtering=False, filter_plan=None)
    result = normalize_intent(
        intent,
        "just show Transaction Date",
        available_columns=["Transaction_Date", "Amount"],
    )
    assert result.needs_filtering is True
    assert "Transaction_Date" in result.filter_plan.select_columns


def test_normalizer_preserves_existing_filter_conditions():
    from finflow_agent.operations.schemas import FilterOperationPlan, FilterCondition
    plan = FilterOperationPlan(
        conditions=[FilterCondition(column="age", operator="gt", value=25)],
        logic="and",
    )
    intent = _base_intent(needs_filtering=True, filter_plan=plan)
    result = normalize_intent(
        intent,
        "only return Customer Id",
        available_columns=["Customer_ID", "age"],
    )
    # Should add select_columns while preserving existing conditions
    assert result.needs_filtering is True
    assert len(result.filter_plan.conditions) == 1
    assert result.filter_plan.conditions[0].column == "age"
    assert "Customer_ID" in result.filter_plan.select_columns
