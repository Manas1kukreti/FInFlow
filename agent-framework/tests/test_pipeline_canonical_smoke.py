import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


from finflow_agent.api import JobPayload, app, handle_upload, process_job_task
from finflow_agent.bootstrap import bootstrap_agents
from finflow_agent.jobs.repository import JobRepository
from finflow_agent.planning.validators import validate_plan
from finflow_agent.state import ExecutionPlan, PlanStep


def test_stage_order_violation_reports_current_message():
    bootstrap_agents()
    plan = ExecutionPlan(
        steps=[
            PlanStep(step_id="step_1", agent="reporting_agent", depends_on=[]),
            PlanStep(
                step_id="step_2",
                agent="ingestion_agent",
                depends_on=["step_1"],
            ),
        ]
    )

    is_valid, err = validate_plan(plan)

    assert not is_valid
    assert "depends on 'step_1' which is at a later stage" in err


@pytest.mark.anyio
async def test_handle_upload_excludes_none_fields_from_enqueue_payload(monkeypatch, tmp_path):
    repo = JobRepository(db_path=str(tmp_path / "jobs.json"))
    redis = AsyncMock()
    app.state.redis = redis

    monkeypatch.setattr("finflow_agent.api.JobRepository", lambda: repo)

    payload = JobPayload(
        submission_id="enq_test_123",
        file_id="safe.csv",
        file_name="safe.csv",
        instruction="clean",
        output_format="csv",
    )

    result = await handle_upload(payload)

    assert result["status"] == "queued"
    assert result["job_id"] == "agent:enq_test_123"
    redis.enqueue_job.assert_called_once_with(
        "process_job_task",
        payload.model_dump(exclude_none=True),
        _job_id="agent:enq_test_123",
    )


@pytest.mark.anyio
async def test_process_job_task_quarantines_legacy_payload(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "safe.csv").write_text("A,B\n1,2\n")

    repo = JobRepository(db_path=str(tmp_path / "jobs.json"))
    payload = {
        "submission_id": "legacy_123",
        "file_id": "safe.csv",
        "file_name": "safe.csv",
        "output_format": "csv",
    }
    await repo.create_or_update_queued("agent:legacy_123", "legacy_123", payload)

    callback = AsyncMock()
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(
        "finflow_agent.jobs.callbacks.send_backend_callback",
        callback,
    )

    await process_job_task({"repository": repo}, payload)

    job = await repo.get_job("agent:legacy_123")

    assert job is not None
    assert job["status"] == "QUARANTINED"
    assert "canonical_intent is required" in job["error"]
    callback.assert_awaited_once()
