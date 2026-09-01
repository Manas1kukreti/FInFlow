from types import SimpleNamespace
from datetime import datetime

import pytest


@pytest.mark.asyncio
async def test_worker_startup_bootstraps_agents(monkeypatch):
    import finflow_agent.api as api

    ctx = {}

    assert hasattr(api, "worker_startup"), (
        "api.py should define worker_startup(ctx) for ARQ workers."
    )

    await api.worker_startup(ctx)

    assert "repository" in ctx
    assert "file_store" in ctx

    from finflow_agent.registry import registry

    for agent_name in [
        "ingestion_agent",
        "cleaning_agent",
        "filter_agent",
        "calculation_agent",
        "reporting_agent",
    ]:
        assert registry.get_spec(agent_name).name == agent_name


@pytest.mark.asyncio
async def test_api_upload_enqueues_arq_job(monkeypatch):
    import finflow_agent.api as api

    calls = {"queued_payload": None, "job_id": None, "stored": None}

    class FakeRedis:
        async def enqueue_job(self, function_name, payload, _job_id=None):
            calls["queued_payload"] = payload
            calls["job_id"] = _job_id
            return SimpleNamespace(job_id=_job_id)

    class FakeRepository:
        async def get_job(self, job_id):
            return None

        async def create_or_update_queued(self, job_id, submission_id, payload):
            calls["stored"] = {
                "job_id": job_id,
                "submission_id": submission_id,
                "payload": payload,
            }

    monkeypatch.setattr(api, "JobRepository", lambda: FakeRepository())
    api.app.state.redis = FakeRedis()

    payload = api.JobPayload(
        submission_id="sub123",
        file_id="input.csv",
        file_name="input.csv",
        instruction="make a report",
        output_format="xlsx",
    )

    response = await api.handle_upload(payload)

    assert response["status"] == "queued"
    assert response["job_id"] == "agent:sub123"
    assert calls["job_id"] == "agent:sub123"
    assert calls["queued_payload"]["file_id"] == "input.csv"
    assert "file_path" not in calls["queued_payload"]
    assert "resolved_file_path" not in calls["queued_payload"]
    assert "canonical_intent" not in calls["queued_payload"]
    assert calls["stored"]["submission_id"] == "sub123"


def test_jobpayload_uses_file_id_not_file_path():
    from finflow_agent.api import JobPayload

    payload = JobPayload(
        submission_id="sub1",
        file_id="input.csv",
        file_name="input.csv",
        instruction="report",
        output_format="xlsx",
    )

    dumped = payload.model_dump()
    assert dumped["file_id"] == "input.csv"
    assert "file_path" not in dumped
    assert "resolved_file_path" in dumped
    assert dumped["resolved_file_path"] is None
    assert "canonical_intent" in dumped
    assert dumped["canonical_intent"] is None


@pytest.mark.asyncio
async def test_engine_failure_summary_preserved(monkeypatch, tmp_path):
    import finflow_agent.api as api
    from finflow_agent.planning.canonical_intent import (
        CanonicalIntent,
        CanonicalIntentEnvelope,
        ProjectColumnsIntent,
        UnresolvedColumnReference,
    )

    stored = {"failed_error": None, "callback_payload": None}

    class FakeRepository:
        async def mark_planning(self, job_id):
            pass

        async def mark_running(self, job_id):
            pass

        async def mark_failed(self, job_id, error_msg):
            stored["failed_error"] = error_msg

        async def mark_succeeded(self, job_id, result):
            raise AssertionError("Should not mark succeeded for failed engine result")

        async def mark_quarantined(self, job_id, reason):
            raise AssertionError("Should not quarantine in this test")

        async def mark_callback_failed(self, job_id, error_msg=None):
            pass

    class FakeFileStore:
        def resolve_uploaded_file(self, file_id):
            path = tmp_path / file_id
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            return path

    class FakeEngine:
        def execute(self, plan, submission_id=None):
            return {
                "status": "failed",
                "output_path": None,
                "summary": {
                    "failed_step_id": "calculate",
                    "error": "Missing required columns in dataset: ['amount']",
                },
            }

    async def fake_callback(payload, job_id, repository):
        stored["callback_payload"] = payload

    monkeypatch.setattr(api, "JobRepository", lambda: FakeRepository())
    monkeypatch.setattr(api, "FileStore", lambda: FakeFileStore())
    monkeypatch.setattr(api, "ExecutionEngine", lambda: FakeEngine())
    monkeypatch.setattr(
        api,
        "compile_canonical_intent",
        lambda intent, **kwargs: object(),
    )

    import finflow_agent.jobs.callbacks as callbacks
    monkeypatch.setattr(callbacks, "send_backend_callback", fake_callback)

    input_file = tmp_path / "input.csv"
    input_file.write_text("Customer_ID,Customer_Name\n1002,Alice\n", encoding="utf-8")

    await api.process_job_task(
        {},
        {
            "submission_id": "sub-fail",
            "file_id": "input.csv",
            "file_name": "input.csv",
            "resolved_file_path": str(input_file),
            "canonical_intent": CanonicalIntentEnvelope(
                schema_version="1.0",
                created_at=datetime.utcnow(),
                original_instruction="return only customer id",
                intent=CanonicalIntent(
                    schema_version="2.0",
                    original_prompt="return only customer id",
                    normalized_prompt="return only customer id",
                    resolution_status="resolved",
                    decision="project_columns",
                    actions=[
                        ProjectColumnsIntent(
                            kind="project_columns",
                            requested_fields=[
                                UnresolvedColumnReference(
                                    raw_reference="customer id",
                                    resolved_column="Customer_ID",
                                    resolution_method="exact_name",
                                )
                            ],
                        )
                    ],
                    output_format="xlsx",
                    assumptions=[],
                    repair_notes=[],
                    dataframe_profile={"source_columns": ["Customer_ID", "Customer_Name"]},
                ),
            ).model_dump(mode="json"),
            "output_format": "xlsx",
            "audit_context": {
                "original_instruction": "delete all rows, calculate loans, and create a chart"
            },
        },
    )

    assert "Step 'calculate' failed:" in stored["failed_error"]
    assert "Missing required columns" in stored["failed_error"]
    assert stored["callback_payload"]["status"] == "failed"
    assert "Missing required columns" in str(stored["callback_payload"]["summary"])


@pytest.mark.asyncio
async def test_process_job_task_uses_canonical_intent_without_legacy_planner(monkeypatch, tmp_path):
    import finflow_agent.api as api
    from finflow_agent.planning.canonical_intent import (
        CanonicalIntent,
        CanonicalIntentEnvelope,
        ProjectColumnsIntent,
        UnresolvedColumnReference,
    )

    stored = {"compiled": None, "callback": None, "marked": []}

    class FakeRepository:
        async def mark_planning(self, job_id):
            stored["marked"].append(("planning", job_id))

        async def mark_running(self, job_id):
            stored["marked"].append(("running", job_id))

        async def mark_failed(self, job_id, error_msg):
            stored["marked"].append(("failed", error_msg))

        async def mark_succeeded(self, job_id, result):
            stored["marked"].append(("succeeded", job_id))

        async def mark_quarantined(self, job_id, reason):
            stored["marked"].append(("quarantined", reason))

        async def mark_callback_failed(self, job_id, error_msg=None):
            stored["marked"].append(("callback_failed", error_msg))

    class FakeFileStore:
        def resolve_uploaded_file(self, file_id):
            raise AssertionError("legacy file-store resolution should not run on canonical jobs")

    class FakeEngine:
        def execute(self, plan, submission_id=None):
            stored["compiled"] = plan
            return {
                "status": "complete",
                "output_path": str(tmp_path / "report.xlsx"),
                "summary": {"ok": True},
            }

    async def fake_callback(payload, job_id, repository):
        stored["callback"] = payload

    def fake_compile(intent, **kwargs):
        stored["compiled_intent"] = intent.model_dump(mode="json")
        stored["compile_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(api, "JobRepository", lambda: FakeRepository())
    monkeypatch.setattr(api, "FileStore", lambda: FakeFileStore())
    monkeypatch.setattr(api, "ExecutionEngine", lambda: FakeEngine())
    monkeypatch.setattr(api, "compile_canonical_intent", fake_compile)

    import finflow_agent.jobs.callbacks as callbacks
    monkeypatch.setattr(callbacks, "send_backend_callback", fake_callback)

    input_file = tmp_path / "input.csv"
    input_file.write_text("Customer_ID,Customer_Name\n1002,Alice\n", encoding="utf-8")

    payload = {
        "submission_id": "sub-canonical",
        "file_id": "input.csv",
        "file_name": "input.csv",
        "resolved_file_path": str(input_file),
        "audit_context": {
            "original_instruction": "delete all rows, calculate loans, and create a chart",
        },
        "canonical_intent": CanonicalIntentEnvelope(
            schema_version="1.0",
            created_at=datetime.utcnow(),
            original_instruction="return only customer id",
            intent=CanonicalIntent(
                schema_version="2.0",
                original_prompt="return only customer id",
                normalized_prompt="return only customer id",
                resolution_status="resolved",
                decision="project_columns",
                actions=[
                    ProjectColumnsIntent(
                        kind="project_columns",
                        requested_fields=[
                            UnresolvedColumnReference(
                                raw_reference="customer id",
                                resolved_column="Customer_ID",
                                resolution_method="exact_name",
                            )
                        ],
                    )
                ],
                output_format="xlsx",
                assumptions=[],
                repair_notes=[],
                dataframe_profile={"source_columns": ["Customer_ID", "Customer_Name"]},
            ),
        ).model_dump(mode="json"),
        "output_format": "xlsx",
    }

    await api.process_job_task({"repository": FakeRepository()}, payload)

    assert stored["compiled_intent"]["actions"][0]["kind"] == "project_columns"
    assert stored["compile_kwargs"]["resolved_file_path"] == str(input_file)
    assert stored["compile_kwargs"]["artifact_prefix"].startswith("submission_sub-canonical_")
    assert stored["compiled"] is not None
    assert stored["callback"]["status"] == "complete"
    assert any(kind == "planning" for kind, _ in stored["marked"])
    assert "audit_context" not in stored["compile_kwargs"]


@pytest.mark.asyncio
async def test_process_job_task_rejects_legacy_only_payload(monkeypatch, tmp_path):
    import finflow_agent.api as api

    stored = {"quarantined": None, "callback": None}

    class FakeRepository:
        async def mark_planning(self, job_id):
            pass

        async def mark_running(self, job_id):
            raise AssertionError("legacy payload should not reach execution")

        async def mark_failed(self, job_id, error_msg):
            raise AssertionError("legacy payload should not reach execution")

        async def mark_succeeded(self, job_id, result):
            raise AssertionError("legacy payload should not reach execution")

        async def mark_quarantined(self, job_id, reason):
            stored["quarantined"] = reason

        async def mark_callback_failed(self, job_id, error_msg=None):
            pass

    class FakeFileStore:
        def resolve_uploaded_file(self, file_id):
            raise AssertionError("legacy file-store resolution should not run")

    async def fake_callback(payload, job_id, repository):
        stored["callback"] = payload

    monkeypatch.setattr(api, "JobRepository", lambda: FakeRepository())
    monkeypatch.setattr(api, "FileStore", lambda: FakeFileStore())

    import finflow_agent.jobs.callbacks as callbacks
    monkeypatch.setattr(callbacks, "send_backend_callback", fake_callback)

    input_file = tmp_path / "input.csv"
    input_file.write_text("Customer_ID,Customer_Name\n1002,Alice\n", encoding="utf-8")

    await api.process_job_task(
        {"repository": FakeRepository()},
        {
            "submission_id": "sub-legacy",
            "file_id": "input.csv",
            "file_name": "input.csv",
            "resolved_file_path": str(input_file),
            "instruction": "return only customer id",
            "output_format": "xlsx",
        },
    )

    assert stored["quarantined"] and "legacy_payload_not_supported" in stored["quarantined"]
    assert stored["callback"]["status"] == "quarantined"
    assert "legacy_payload_not_supported" in stored["callback"]["summary"]["reason"]


@pytest.mark.asyncio
async def test_process_job_task_rejects_invalid_canonical_schema_version(monkeypatch, tmp_path):
    import finflow_agent.api as api
    from finflow_agent.planning.canonical_intent import (
        CanonicalIntent,
        CanonicalIntentEnvelope,
        ProjectColumnsIntent,
        UnresolvedColumnReference,
    )

    stored = {"quarantined": None, "callback": None}

    class FakeRepository:
        async def mark_planning(self, job_id):
            pass

        async def mark_running(self, job_id):
            raise AssertionError("invalid canonical intent should not reach execution")

        async def mark_failed(self, job_id, error_msg):
            raise AssertionError("invalid canonical intent should not execute")

        async def mark_succeeded(self, job_id, result):
            raise AssertionError("invalid canonical intent should not succeed")

        async def mark_quarantined(self, job_id, reason):
            stored["quarantined"] = reason

        async def mark_callback_failed(self, job_id, error_msg=None):
            pass

    class FakeFileStore:
        def resolve_uploaded_file(self, file_id):
            raise AssertionError("legacy file-store resolution should not run on canonical jobs")

    async def fake_callback(payload, job_id, repository):
        stored["callback"] = payload

    monkeypatch.setattr(api, "JobRepository", lambda: FakeRepository())
    monkeypatch.setattr(api, "FileStore", lambda: FakeFileStore())

    import finflow_agent.jobs.callbacks as callbacks
    monkeypatch.setattr(callbacks, "send_backend_callback", fake_callback)

    input_file = tmp_path / "input.csv"
    input_file.write_text("Customer_ID,Customer_Name\n1002,Alice\n", encoding="utf-8")

    payload = {
        "submission_id": "sub-invalid-version",
        "file_id": "input.csv",
        "file_name": "input.csv",
        "resolved_file_path": str(input_file),
        "canonical_intent": CanonicalIntentEnvelope(
            schema_version="1.0",
            created_at=datetime.utcnow(),
            original_instruction="return only customer id",
            intent=CanonicalIntent(
                schema_version="9.9",
                original_prompt="return only customer id",
                normalized_prompt="return only customer id",
                resolution_status="resolved",
                decision="project_columns",
                actions=[
                    ProjectColumnsIntent(
                        kind="project_columns",
                        requested_fields=[
                            UnresolvedColumnReference(
                                raw_reference="customer id",
                                resolved_column="Customer_ID",
                                resolution_method="exact_name",
                            )
                        ],
                    )
                ],
                output_format="xlsx",
                assumptions=[],
                repair_notes=[],
                dataframe_profile={"source_columns": ["Customer_ID", "Customer_Name"]},
            ),
        ).model_dump(mode="json"),
        "output_format": "xlsx",
    }

    await api.process_job_task({"repository": FakeRepository()}, payload)

    assert "Unsupported canonical intent schema_version" in stored["quarantined"]
    assert stored["callback"]["status"] == "quarantined"
