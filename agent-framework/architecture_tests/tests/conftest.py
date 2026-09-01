import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add the src directory to sys.path so finflow_agent is importable
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def stable_env(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("BACKEND_CALLBACK_URL", "http://backend.test/api/agent/callback")
    monkeypatch.setenv("AGENT_CALLBACK_SECRET", "test-secret")

    yield


@pytest.fixture
def bootstrap_agents():
    from finflow_agent.bootstrap import bootstrap_agents, validate_required_agents_registered

    bootstrap_agents()
    validate_required_agents_registered()
