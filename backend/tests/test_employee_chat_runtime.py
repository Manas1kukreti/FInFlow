from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from app.models import User, UserRole
from app.models.chat import ChatDataset
from app.services.chat_authorization import authorize_chat_intent
from app.services.chat_intent import ChatCanonicalIntent, ChatDatasetReference, ChatFieldReference, ChatOperation
from app.services.chat_runtime import build_chat_result


def _dataset() -> ChatDataset:
    return ChatDataset(
        id=uuid4(),
        dataset_key="employee_directory",
        display_name="Employee Directory",
        description="Employee metrics",
        domain="people",
        physical_source="chat_dataset_rows",
        storage_location="chat_dataset_rows.payload",
        schema_version="1.0",
        columns_json=[
            {
                "physical_column": "department",
                "semantic_name": "department",
                "description": "Department",
                "data_type": "string",
                "sample_values": ["Finance", "Technology"],
                "synonyms": ["team"],
                "sensitivity": "internal",
            },
            {
                "physical_column": "salary",
                "semantic_name": "salary",
                "description": "Salary",
                "data_type": "number",
                "sample_values": [100, 200],
                "synonyms": ["pay"],
                "sensitivity": "restricted",
            },
        ],
        aliases=["employees"],
        sample_values_json={},
        owner="People Ops",
        sensitivity="internal",
        allowed_roles=["employee", "manager", "admin"],
        row_level_policy_json={},
        column_level_policy_json={},
        chat_enabled=True,
        last_updated_at=datetime.now(UTC),
    )


def test_build_chat_result_returns_scalar_count():
    dataset = _dataset()
    frame = pd.DataFrame(
        [
            {"department": "Finance", "salary": 100},
            {"department": "Finance", "salary": 200},
            {"department": "Technology", "salary": 300},
        ]
    )
    intent = {
        "intent_type": "aggregate_query",
        "operations": [{"operation": "count"}],
        "filters": [
            {
                "field": {"resolved_column": "department", "raw_reference": "department"},
                "operator": "equals",
                "resolved_value": "Finance",
                "raw_value": "Finance",
            }
        ],
        "group_by": [],
        "visualization": None,
    }

    result = build_chat_result(execution_id=uuid4(), dataset=dataset, intent=intent, frame=frame)

    assert result["result_type"] == "scalar"
    assert result["value"] == 2
    assert result["formatted_value"] == "2"
    assert "Finance" in result["summary"]


def test_build_chat_result_returns_grouped_chart():
    dataset = _dataset()
    frame = pd.DataFrame(
        [
            {"department": "Finance", "salary": 100},
            {"department": "Finance", "salary": 200},
            {"department": "Technology", "salary": 300},
        ]
    )
    intent = {
        "intent_type": "grouped_analysis",
        "operations": [{"operation": "count"}],
        "filters": [],
        "group_by": [
            {
                "resolved_column": "department",
                "raw_reference": "department",
            }
        ],
        "visualization": {"chart_type": "bar", "title": "Headcount by department"},
    }

    result = build_chat_result(execution_id=uuid4(), dataset=dataset, intent=intent, frame=frame)

    assert result["result_type"] == "chart"
    assert result["chart"]["chart_type"] == "bar"
    assert result["chart"]["encoding"]["x"] == "department"
    assert result["chart"]["encoding"]["y"] == "count"
    assert len(result["rows"]) == 2


def test_authorize_chat_intent_denies_employee_record_lookup_for_salary():
    dataset = _dataset()
    intent = ChatCanonicalIntent(
        intent_type="record_lookup",
        dataset_reference=ChatDatasetReference(raw_reference="employee_directory", resolved_dataset_id=str(dataset.id), resolved_dataset_key=dataset.dataset_key),
        operations=[
            ChatOperation(
                operation="count",
                metric=ChatFieldReference(raw_reference="salary", resolved_column="salary"),
            )
        ],
    )
    user = User(
        id=uuid4(),
        full_name="Employee",
        email="employee@example.com",
        hashed_password="hashed",
        role=UserRole.employee,
    )

    decision = authorize_chat_intent(user, dataset, intent)

    assert decision.decision == "denied"
    assert decision.reason_code == "COLUMN_ACCESS_RESTRICTED"
    assert "salary" in decision.restricted_columns

