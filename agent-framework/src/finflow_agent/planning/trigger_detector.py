"""Trigger detection for visualization trigger language.

Detects explicit visualization requests in user prompts using
case-insensitive, whole-word boundary matching. Substrings within
larger words (e.g., "chart" in "uncharted") do not constitute a match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TriggerResult:
    """Result of trigger detection on a user prompt."""

    triggered: bool
    matched_term: str | None = None
    chart_type_hint: str | None = None


# Single-word trigger terms that activate visualization.
TRIGGER_TERMS: set[str] = {
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualise",
    "visualization",
    "visualisation",
}

# Multi-word phrases that activate visualization.
# Order matters: longer/more-specific phrases are checked first.
TRIGGER_PHRASES: list[str] = [
    "pie chart",
    "bar chart",
    "line chart",
    "scatter plot",
    "histogram",
    "as a chart",
    "as a graph",
]

# Analytical terms that do NOT trigger visualization on their own.
ANALYTICAL_ONLY_TERMS: set[str] = {
    "trend",
    "distribution",
    "compare",
    "breakdown",
    "summary",
    "overview",
    "analysis",
}

# Mapping from trigger phrases to chart type hints.
_PHRASE_TO_CHART_TYPE: dict[str, str] = {
    "pie chart": "pie",
    "bar chart": "bar",
    "line chart": "line",
    "scatter plot": "scatter",
    "histogram": "histogram",
}


def _build_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Build a regex pattern for an exact contiguous phrase with word boundaries."""
    escaped = re.escape(phrase)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _build_term_pattern(term: str) -> re.Pattern[str]:
    """Build a regex pattern for a single term with word boundaries."""
    escaped = re.escape(term)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


# Pre-compiled patterns for phrases (checked first, longest match priority).
_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (phrase, _build_phrase_pattern(phrase)) for phrase in TRIGGER_PHRASES
]

# Pre-compiled patterns for single terms.
_TERM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (term, _build_term_pattern(term)) for term in TRIGGER_TERMS
]


class TriggerDetector:
    """Detects visualization trigger language in user prompts.

    Uses case-insensitive, whole-word boundary matching to identify
    explicit visualization requests. Trigger language takes precedence
    over analytical-only terms.
    """

    def detect(self, prompt: str) -> TriggerResult:
        """Detect visualization trigger language in a user prompt.

        Args:
            prompt: The user's input prompt text.

        Returns:
            TriggerResult indicating whether visualization was triggered,
            the matched term/phrase, and an optional chart type hint.
        """
        if not prompt or not prompt.strip():
            return TriggerResult(triggered=False)

        # Check multi-word phrases first (more specific matches take priority).
        for phrase, pattern in _PHRASE_PATTERNS:
            if pattern.search(prompt):
                chart_type_hint = _PHRASE_TO_CHART_TYPE.get(phrase)
                return TriggerResult(
                    triggered=True,
                    matched_term=phrase,
                    chart_type_hint=chart_type_hint,
                )

        # Check single-word trigger terms.
        for term, pattern in _TERM_PATTERNS:
            if pattern.search(prompt):
                return TriggerResult(
                    triggered=True,
                    matched_term=term,
                    chart_type_hint=None,
                )

        # No trigger language found.
        return TriggerResult(triggered=False)
