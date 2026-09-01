from __future__ import annotations

import pandas as pd

from finflow_agent.contract_registry import CanonicalOperator
from finflow_agent.tools.value_resolver import resolve_value_domain


def test_resolve_value_domain_accepts_canonical_operator_enum_eq():
    series = pd.Series(["Female", "Male", "Female"], name="gender")

    result = resolve_value_domain(series, "Female", CanonicalOperator.EQ)

    assert result.operator == "eq"
    assert result.matched is True
    assert result.match_kind in {"exact", "normalized"}
    assert result.match_count == 1


def test_resolve_value_domain_accepts_canonical_operator_enum_contains():
    series = pd.Series(["female applicant", "male applicant"], name="description")

    result = resolve_value_domain(series, "female", CanonicalOperator.CONTAINS)

    assert result.operator == "contains"
    assert result.matched is True
    assert result.match_kind in {"exact", "normalized"}
