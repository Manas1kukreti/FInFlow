import asyncio
import sys
from pathlib import Path

import pytest


SRC_ROOT = (
    Path(__file__).resolve().parents[2]
    / "agent-framework"
    / "src"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finflow_agent.jobs.callbacks import send_backend_callback


class DummyRepository:
    def __init__(self):
        self.callback_failed = False
        self.callback_error = None

    async def mark_callback_failed(self, job_id: str, error_msg: str | None = None) -> None:
        self.callback_failed = True
        self.callback_error = error_msg


def test_send_backend_callback_requires_explicit_secret(monkeypatch):
    async def run() -> None:
        monkeypatch.delenv("AGENT_CALLBACK_SECRET", raising=False)
        monkeypatch.setenv(
            "BACKEND_CALLBACK_URL", "http://backend.test/api/agent/callback"
        )

        repository = DummyRepository()

        with pytest.raises(RuntimeError, match="AGENT_CALLBACK_SECRET is required"):
            await send_backend_callback(
                {"status": "complete", "summary": {"ok": True}},
                "job-1",
                repository,
            )

        assert repository.callback_failed is False

    asyncio.run(run())
