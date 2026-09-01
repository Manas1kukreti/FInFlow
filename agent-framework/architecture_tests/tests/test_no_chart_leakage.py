"""Deterministic chart-leakage audit (task 13.2).

Locks the spec contract from Requirements 9.4 and 9.5: the cleaning,
filter, and reporting agents — and their handlers in ``operations/`` — MUST
NOT generate, render, or embed any chart in any output artifact. Visualization
is disabled by default in this version (the visualization scaffolding lives
exclusively in ``agents/visualization_agent.py`` and its handler module, both
of which are out of scope for this audit).

Audit rules (per task 13.2):

* Each file's source is read and tokenized via the standard library's
  :mod:`tokenize` module so that COMMENT tokens are stripped before any
  forbidden-token / forbidden-import scan. Strings (including docstrings)
  are intentionally NOT stripped — a forbidden symbol smuggled into a
  string literal would still ship in production.
* Forbidden imports (regex anchored to the start of a line):
  ``matplotlib``, ``plotly``, ``seaborn``. Agent files (not handlers) also
  forbid ``xlsxwriter`` — chart rendering only happens through xlsxwriter
  and the agents themselves must never import it directly. The audit-sheet
  writer in ``operations/reporting_handlers.py`` may use ``xlsxwriter``
  because it produces tabular sheets only.
* Forbidden symbol substrings (after comment stripping):
  ``add_chart(``, ``insert_chart(``, ``pyplot``, ``savefig``,
  ``matplotlib``, ``plotly``, ``seaborn``.
* Bare symbol ``Chart`` (whole-word match) — but ``ChartSpec``,
  ``ChartArtifact``, and ``ChartConfig`` are explicitly allowed because
  they are the visualization-plan model identifiers (the scaffolded /
  disabled types, not chart-rendering calls). The regex uses ``\\bChart\\(``
  so only the bare constructor invocation is rejected.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path
from typing import List

import pytest


# Project root resolution: this test file lives at
# ``<repo>/finflow_architecture_tests/tests/test_no_chart_leakage.py``.
# The source tree it audits lives at ``<repo>/src/finflow_agent/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "finflow_agent"

_AGENT_FILES: List[Path] = [
    _SRC_ROOT / "agents" / "cleaning_agent.py",
    _SRC_ROOT / "agents" / "filter_agent.py",
    _SRC_ROOT / "agents" / "reporting_agent.py",
]

_HANDLER_FILES: List[Path] = [
    _SRC_ROOT / "operations" / "cleaning_handlers.py",
    _SRC_ROOT / "operations" / "filter_handlers.py",
    _SRC_ROOT / "operations" / "reporting_handlers.py",
    _SRC_ROOT / "operations" / "executor.py",
]


# ---------------------------------------------------------------------------
# Audit primitives
# ---------------------------------------------------------------------------

def _strip_comments(source: str) -> str:
    """Return *source* with every ``# ...`` comment removed.

    Uses :func:`tokenize.generate_tokens` to identify COMMENT tokens and
    rewrites each comment line in place, preserving every other byte
    (whitespace, newlines, code, strings, docstrings) exactly. We do NOT
    feed the result through :func:`tokenize.untokenize` because that
    helper rewrites whitespace and would mask layout-sensitive checks.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        # Defensive: if the file fails to tokenize we keep the raw source
        # so the forbidden-token regex still has something to match. A
        # truly broken file would fail the surrounding test suite anyway.
        return source

    lines = source.splitlines(keepends=True)
    # Process comments back-to-front so column offsets within each line
    # remain valid as we rewrite.
    comments = sorted(
        (tok for tok in tokens if tok.type == tokenize.COMMENT),
        key=lambda t: (t.start[0], t.start[1]),
        reverse=True,
    )
    for tok in comments:
        line_idx = tok.start[0] - 1
        col = tok.start[1]
        if line_idx < 0 or line_idx >= len(lines):
            continue
        line = lines[line_idx]
        # Preserve any trailing newline so subsequent regex anchors
        # (e.g. ``^`` and ``$``) keep working.
        nl_split = re.split(r"(\r\n|\n|\r)", line, maxsplit=1)
        prefix = nl_split[0]
        newline = "".join(nl_split[1:]) if len(nl_split) > 1 else ""
        if col >= len(prefix):
            # Comment at or beyond end-of-line: nothing to strip.
            continue
        lines[line_idx] = prefix[:col].rstrip(" \t") + newline
    return "".join(lines)


def _read_stripped_source(path: Path) -> str:
    """Read *path* as UTF-8 and return its source with comments stripped."""
    if not path.is_file():
        raise FileNotFoundError(f"Audited file is missing: {path}")
    raw = path.read_text(encoding="utf-8")
    return _strip_comments(raw)


# ---------------------------------------------------------------------------
# Forbidden patterns shared by every audited file.
# ---------------------------------------------------------------------------

# Forbidden import lines (top-of-line, ignoring leading whitespace). Each
# pattern matches both ``import X`` and ``from X import ...`` forms.
_FORBIDDEN_IMPORT_PATTERNS = {
    "matplotlib": re.compile(
        r"^\s*(?:import|from)\s+matplotlib\b", re.MULTILINE
    ),
    "plotly": re.compile(
        r"^\s*(?:import|from)\s+plotly\b", re.MULTILINE
    ),
    "seaborn": re.compile(
        r"^\s*(?:import|from)\s+seaborn\b", re.MULTILINE
    ),
}

# Additional import forbidden ONLY in agent files. Handlers may use
# ``xlsxwriter`` for tabular sheet formatting (bold headers, frozen
# panes, autofit) per task 13.2's note 1.
_AGENT_ONLY_FORBIDDEN_IMPORT = {
    "xlsxwriter": re.compile(
        r"^\s*(?:import|from)\s+xlsxwriter\b", re.MULTILINE
    ),
}

# Forbidden symbol substrings. The compiled regexes treat ``add_chart(``
# and ``insert_chart(`` as concrete API calls; the remaining tokens are
# library names whose mere presence (anywhere in stripped source) leaks
# chart intent.
_FORBIDDEN_SYMBOL_PATTERNS = {
    "add_chart(": re.compile(r"add_chart\s*\("),
    "insert_chart(": re.compile(r"insert_chart\s*\("),
    "pyplot": re.compile(r"\bpyplot\b"),
    "savefig": re.compile(r"\bsavefig\b"),
    "matplotlib": re.compile(r"\bmatplotlib\b"),
    "plotly": re.compile(r"\bplotly\b"),
    "seaborn": re.compile(r"\bseaborn\b"),
}

# Bare ``Chart(`` constructor — whole-word match. ``ChartSpec(``,
# ``ChartArtifact(``, and ``ChartConfig(`` are explicitly allowed because
# the regex requires the ``(`` to immediately follow ``Chart``; in those
# identifiers the suffix (``Spec``/``Artifact``/``Config``) sits between
# the ``Chart`` token and the parenthesis, so they do not match.
_BARE_CHART_PATTERN = re.compile(r"\bChart\s*\(")


# ---------------------------------------------------------------------------
# Reusable audit helper
# ---------------------------------------------------------------------------

def _assert_no_chart_leakage(path: Path, *, is_agent: bool) -> None:
    """Run the full audit suite against *path*.

    Parameters
    ----------
    path:
        Absolute path to the source file under audit.
    is_agent:
        ``True`` when *path* lives in ``agents/`` (then ``xlsxwriter``
        imports are also forbidden); ``False`` for handler / executor
        files (which may import ``xlsxwriter`` for tabular sheet output).
    """
    stripped = _read_stripped_source(path)

    # 1. Forbidden imports (always).
    for name, pattern in _FORBIDDEN_IMPORT_PATTERNS.items():
        match = pattern.search(stripped)
        assert match is None, (
            f"{path.name}: forbidden import of {name!r} detected at offset "
            f"{match.start()}: {match.group(0)!r}"
        )

    # 2. Agent-only forbidden imports.
    if is_agent:
        for name, pattern in _AGENT_ONLY_FORBIDDEN_IMPORT.items():
            match = pattern.search(stripped)
            assert match is None, (
                f"{path.name}: agents must not import {name!r}; charts are "
                f"only rendered through xlsxwriter and the agents must stay "
                f"chart-free. Match at offset {match.start()}: "
                f"{match.group(0)!r}"
            )

    # 3. Forbidden symbol substrings.
    for token, pattern in _FORBIDDEN_SYMBOL_PATTERNS.items():
        match = pattern.search(stripped)
        assert match is None, (
            f"{path.name}: forbidden chart-leakage symbol {token!r} found at "
            f"offset {match.start()}: {match.group(0)!r}"
        )

    # 4. Bare ``Chart(`` constructor.
    match = _BARE_CHART_PATTERN.search(stripped)
    assert match is None, (
        f"{path.name}: bare 'Chart(' constructor found at offset "
        f"{match.start()}: {match.group(0)!r}. ChartSpec / ChartArtifact / "
        f"ChartConfig are allowed (they don't match this regex), but a bare "
        f"'Chart(' indicates a chart-rendering call."
    )


# ---------------------------------------------------------------------------
# One named test per audited file (per task 13.2's test-name suggestions)
# ---------------------------------------------------------------------------

def test_no_chart_imports_in_cleaning_agent() -> None:
    """**Validates: Requirements 9.4, 9.5**

    Cleaning_Agent must never generate, render, or embed any chart.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "agents" / "cleaning_agent.py", is_agent=True
    )


def test_no_chart_imports_in_filter_agent() -> None:
    """**Validates: Requirements 9.4, 9.5**

    Filter_Agent must never generate, render, or embed any chart.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "agents" / "filter_agent.py", is_agent=True
    )


def test_no_chart_imports_in_reporting_agent() -> None:
    """**Validates: Requirements 9.4, 9.5**

    Reporting_Agent must never generate, render, or embed any chart in
    the output Excel file.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "agents" / "reporting_agent.py", is_agent=True
    )


def test_no_chart_imports_in_cleaning_handlers() -> None:
    """**Validates: Requirements 9.4, 9.5**

    Cleaning handlers must stay chart-free; cleaning is purely tabular.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "operations" / "cleaning_handlers.py", is_agent=False
    )


def test_no_chart_imports_in_filter_handlers() -> None:
    """**Validates: Requirements 9.4, 9.5**

    Filter handlers must stay chart-free; filtering is purely tabular.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "operations" / "filter_handlers.py", is_agent=False
    )


def test_no_chart_calls_in_reporting_handlers() -> None:
    """**Validates: Requirements 9.4, 9.5**

    The reporting handlers (audit-sheet writer + back-compat csv/json/txt
    writers) may import ``xlsxwriter`` for tabular sheet formatting but
    MUST NOT call ``add_chart`` / ``insert_chart`` or otherwise embed a
    chart in any sheet.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "operations" / "reporting_handlers.py", is_agent=False
    )


def test_no_chart_imports_in_executor() -> None:
    """**Validates: Requirements 9.4, 9.5**

    The deterministic executor dispatches tabular operations only; it
    must never reference any chart-rendering library or API.
    """
    _assert_no_chart_leakage(
        _SRC_ROOT / "operations" / "executor.py", is_agent=False
    )


# ---------------------------------------------------------------------------
# Self-test for the comment stripper. Locking the contract that comment
# bodies are removed (so a forbidden token inside a ``#`` comment would
# NOT trip the audit), while string literals are preserved (so a smuggled
# forbidden token inside a string literal WOULD trip the audit).
# ---------------------------------------------------------------------------

def test_strip_comments_removes_only_comment_tokens() -> None:
    """The comment stripper must drop ``# ...`` tokens but leave string
    literals (and docstrings) intact, so a forbidden token smuggled into
    a string literal still trips the audit."""
    source = (
        '"""Docstring referencing matplotlib"""\n'
        "import os  # this comment mentions add_chart(\n"
        "x = 'plotly inside string'\n"
        "# top-level comment with seaborn\n"
        "y = 1\n"
    )
    stripped = _strip_comments(source)

    # Comment bodies are removed.
    assert "this comment mentions add_chart(" not in stripped
    assert "top-level comment with seaborn" not in stripped

    # String / docstring contents survive (so the audit can still see
    # smuggled tokens).
    assert "matplotlib" in stripped
    assert "plotly inside string" in stripped

    # Code outside comments is preserved.
    assert "import os" in stripped
    assert "y = 1" in stripped
