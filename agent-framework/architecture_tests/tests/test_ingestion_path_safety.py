"""Tests for ingestion path safety.

These tests pin down two contracts:

1. ``get_safe_input_path(base_dir, file_path)`` rejects every path that is not a
   regular file inside ``base_dir`` (traversal, absolute paths outside the
   sandbox, missing files, directories) and returns the resolved ``Path`` on
   success.
2. ``IngestionAgent.execute`` wires that helper in when ``UPLOAD_DIR`` is set,
   so a malicious or buggy compiler/orchestrator that hands the agent a
   ``resolved_file_path`` outside the configured upload directory produces an
   ``AgentResult(status="failed", ...)`` instead of being silently read.

The conftest fixture ``stable_env`` is autouse and sets ``UPLOAD_DIR`` to
``tmp_path / "uploads"``. Tests that need the back-compat behaviour
(``UPLOAD_DIR`` unset) explicitly delete it via ``monkeypatch.delenv``.
"""
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# get_safe_input_path
# ---------------------------------------------------------------------------


def test_get_safe_input_path_accepts_file_inside_base_dir(tmp_path):
    from finflow_agent.tools.path_safety import get_safe_input_path

    base_dir = tmp_path / "uploads"
    base_dir.mkdir(exist_ok=True)
    safe_file = base_dir / "input.csv"
    safe_file.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = get_safe_input_path(str(base_dir), str(safe_file))

    assert isinstance(resolved, Path)
    assert resolved.exists()
    assert resolved.is_file()
    assert resolved.resolve() == safe_file.resolve()


def test_get_safe_input_path_rejects_traversal(tmp_path):
    from finflow_agent.tools.path_safety import get_safe_input_path
    from finflow_agent.operations.errors import UnsafeInputPathError

    base_dir = tmp_path / "uploads"
    base_dir.mkdir(exist_ok=True)

    # Place a real file outside the base dir so the only thing keeping the
    # caller out is the sandbox check, not an "exists" check.
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(UnsafeInputPathError):
        get_safe_input_path(str(base_dir), "../outside.csv")


def test_get_safe_input_path_rejects_absolute_outside_base(tmp_path):
    from finflow_agent.tools.path_safety import get_safe_input_path
    from finflow_agent.operations.errors import UnsafeInputPathError

    base_dir = tmp_path / "uploads"
    base_dir.mkdir(exist_ok=True)

    # Build a real, existing file that is platform-agnostically outside the
    # base dir. Using a sibling directory under tmp_path keeps the test
    # deterministic on Windows, Linux, and macOS.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.csv"
    outside_file.write_text("x,y\n9,9\n", encoding="utf-8")

    with pytest.raises(UnsafeInputPathError):
        get_safe_input_path(str(base_dir), str(outside_file))


def test_get_safe_input_path_rejects_nonexistent_file(tmp_path):
    from finflow_agent.tools.path_safety import get_safe_input_path
    from finflow_agent.operations.errors import UnsafeInputPathError

    base_dir = tmp_path / "uploads"
    base_dir.mkdir(exist_ok=True)
    missing = base_dir / "nope.csv"

    with pytest.raises(UnsafeInputPathError):
        get_safe_input_path(str(base_dir), str(missing))


def test_get_safe_input_path_rejects_directory(tmp_path):
    from finflow_agent.tools.path_safety import get_safe_input_path
    from finflow_agent.operations.errors import UnsafeInputPathError

    base_dir = tmp_path / "uploads"
    base_dir.mkdir(exist_ok=True)
    nested_dir = base_dir / "nested"
    nested_dir.mkdir(exist_ok=True)

    with pytest.raises(UnsafeInputPathError):
        get_safe_input_path(str(base_dir), str(nested_dir))


# ---------------------------------------------------------------------------
# IngestionAgent.execute integration
# ---------------------------------------------------------------------------


def test_ingestion_agent_rejects_traversal_input_path(tmp_path, monkeypatch):
    """A compiled plan with a traversal-laced ``resolved_file_path`` must fail
    cleanly with an unsafe-path error, not read the file outside the sandbox.
    """
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

    # Real file outside the upload sandbox.
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("a,b\n1,2\n", encoding="utf-8")

    # Traversal segment: relative path that escapes the upload dir.
    traversal_path = "../outside.csv"

    result = IngestionAgent().execute(
        {"resolved_file_path": traversal_path, "file_type": "csv"},
        {},
    )

    assert result.status == "failed"
    assert result.error_message is not None
    assert "unsafe input path" in result.error_message.lower()


def test_ingestion_agent_rejects_absolute_path_outside_upload_dir(tmp_path, monkeypatch):
    """Even an absolute path is rejected when it points outside ``UPLOAD_DIR``."""
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "leak.csv"
    outside_file.write_text("a,b\n1,2\n", encoding="utf-8")

    result = IngestionAgent().execute(
        {"resolved_file_path": str(outside_file), "file_type": "csv"},
        {},
    )

    assert result.status == "failed"
    assert result.error_message is not None
    assert "unsafe input path" in result.error_message.lower()


def test_ingestion_agent_accepts_safe_path_under_upload_dir(tmp_path, monkeypatch):
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

    csv_path = upload_dir / "test.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = IngestionAgent().execute(
        {"resolved_file_path": str(csv_path), "file_type": "csv"},
        {},
    )

    assert result.status == "success", result.error_message
    assert result.metrics.get("row_count") == 2
    assert result.metrics.get("column_count") == 2


def test_ingestion_agent_falls_back_when_upload_dir_unset(tmp_path, monkeypatch):
    """When ``UPLOAD_DIR`` is not configured, the agent falls back to a plain
    existence check so legacy callers keep working.
    """
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    monkeypatch.delenv("UPLOAD_DIR", raising=False)

    csv_path = tmp_path / "legacy_input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = IngestionAgent().execute(
        {"resolved_file_path": str(csv_path), "file_type": "csv"},
        {},
    )

    assert result.status == "success", result.error_message
    assert result.metrics.get("row_count") == 1
    assert result.metrics.get("column_count") == 2


def test_ingestion_agent_parse_error_includes_file_context(tmp_path, monkeypatch):
    from finflow_agent.agents.ingestion_agent import IngestionAgent

    monkeypatch.delenv("UPLOAD_DIR", raising=False)

    missing_excel = tmp_path / "broken.xlsx"
    missing_excel.write_text("not a real excel file", encoding="utf-8")

    result = IngestionAgent().execute(
        {"resolved_file_path": str(missing_excel), "file_type": "xlsx"},
        {},
    )

    assert result.status == "failed"
    assert result.error_message is not None
    assert "failed to parse input file" in result.error_message.lower()
    assert "broken.xlsx" in result.error_message
    assert "file_type='xlsx'" in result.error_message
