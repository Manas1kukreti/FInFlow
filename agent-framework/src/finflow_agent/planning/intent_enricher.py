"""Visualization intent enricher for the canonical intent pipeline.

Integrates the TriggerDetector with the intent classification pipeline.
When trigger language is detected in the user prompt, this module enriches
the canonical intent's actions list with VisualizeIntent actions — one per
detected chart request.

Supports multi-chart prompts: "show a pie chart of X, a bar chart of Y,
and a line chart of Z" produces 3 separate VisualizeIntent actions.

Flow:
    User Prompt → parse_chart_requests() → N chart specs detected?
        YES → Add N VisualizeIntent actions to canonical intent
        NO  → Leave canonical intent unchanged

Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
"""

from __future__ import annotations

import re
from typing import Any

from finflow_agent.planning.trigger_detector import TriggerDetector


# Module-level singleton detector instance.
_detector = TriggerDetector()

# Chart type patterns for multi-chart extraction
_CHART_HEAD_PATTERN = re.compile(
    r"(?:^|(?:\band\b|\bthen\b|\balso\b|[.;]))\s*"
    r"(?:(?:generate|create|show|display|make|build|produce)\s+)?"
    r"(?:(?:a|an|the|single|one)\s+){0,2}"
    r"(?P<type>pie|pi|bar|line|scatter|histogram|grouped\s+bar|clustered\s+bar|stacked\s+bar)\s*"
    r"(?:chart|graph|plot|visualization|diagram)s?\b",
    re.IGNORECASE,
)

_CHART_SEPARATOR = re.compile(
    r"\.\s*(?:next|then|also|finally|additionally)?\s*,?\s*"
    r"|\bnext\b\s*,?\s*"
    r"|\bfinally\b\s*,?\s*"
    r"|\balso\b\s*,?\s*"
    r"|\bthen\b\s*,?\s*",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "the", "a", "an", "of", "for", "with", "and", "showing", "displaying",
    "show", "display", "create", "make", "build", "produce", "chart", "graph",
    "plot", "visualization", "diagram", "single", "clear", "number", "count",
    "counts", "people", "bars", "bar", "axis", "legend", "labels", "label",
    "classification", "classifications", "separate", "side", "by", "split",
}


def _clean_field_phrase(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_ ]+", " ", text).strip().lower()
    words = [word for word in value.split() if word and word not in _STOP_WORDS]
    if not words:
        return ""
    if len(words) >= 2:
        return "_".join(words[:2])
    return words[0]


def _extract_visual_semantics(description: str, chart_type_raw: str) -> dict[str, Any]:
    desc = " ".join(description.split())
    desc_lower = desc.lower()
    chart_type_lower = chart_type_raw.lower()

    chart_type = "pie" if chart_type_lower in ("pi", "pie") else "bar" if "bar" in chart_type_lower else chart_type_lower
    bar_mode = None
    if "grouped" in chart_type_lower or "clustered" in chart_type_lower or "side by side" in desc_lower or "separate" in desc_lower:
        bar_mode = "grouped"
    elif "stacked" in chart_type_lower:
        bar_mode = "stacked"

    aggregation = "count"
    measure = None
    if re.search(r"\b(avg|average|mean)\b", desc_lower):
        aggregation = "mean"
    elif re.search(r"\b(sum|total)\b", desc_lower):
        aggregation = "sum"

    if aggregation in {"mean", "sum"}:
        measure_match = re.search(r"\b(?:avg|average|mean|sum|total)\s+([A-Za-z_][A-Za-z0-9_ ]*)\s+by\b", desc, re.IGNORECASE)
        if measure_match:
            measure = _clean_field_phrase(measure_match.group(1))

    x_field = None
    series_field = None

    split_match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_ ]*?)\s+(?:split by|broken down by|grouped by)\s+([A-Za-z_][A-Za-z0-9_ ]+)",
        desc,
        re.IGNORECASE,
    )
    x_axis_match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_ ]*?)\s+on\s+the\s+x[\s-]?axis\b",
        desc,
        re.IGNORECASE,
    )
    by_match = re.search(
        r"\b(?:count(?: of)?|number of|average|avg|mean|sum|total)?\s*([A-Za-z_][A-Za-z0-9_ ]*?)\s+by\s+([A-Za-z_][A-Za-z0-9_ ]+)",
        desc,
        re.IGNORECASE,
    )
    and_match = re.search(
        r"\bby\s+([A-Za-z_][A-Za-z0-9_ ]*?)\s+(?:and|,)\s+([A-Za-z_][A-Za-z0-9_ ]+)",
        desc,
        re.IGNORECASE,
    )

    if split_match:
        x_field = _clean_field_phrase(split_match.group(1))
        series_field = _clean_field_phrase(split_match.group(2))
    elif and_match:
        x_field = _clean_field_phrase(and_match.group(1))
        series_field = _clean_field_phrase(and_match.group(2))
    elif by_match:
        series_field = _clean_field_phrase(by_match.group(1))
        x_field = _clean_field_phrase(by_match.group(2))

    if x_axis_match:
        explicit_x = _clean_field_phrase(x_axis_match.group(1))
        if explicit_x:
            x_field = explicit_x
            if series_field == explicit_x:
                series_field = None

    if x_field and not series_field:
        by_series_match = re.search(r"\bby\s+([A-Za-z_][A-Za-z0-9_ ]*?)(?:,|\bwith\b|$)", desc, re.IGNORECASE)
        if by_series_match:
            candidate = _clean_field_phrase(by_series_match.group(1))
            if candidate and candidate != x_field:
                series_field = candidate

    if chart_type == "bar" and series_field and bar_mode is None:
        bar_mode = "grouped"

    return {
        "kind": "visualize",
        "chart_type": chart_type,
        "fields": [],
        "description": desc,
        "x": x_field,
        "series": series_field,
        "group_by": [field for field in [x_field, series_field] if field],
        "measure": measure,
        "aggregation": aggregation,
        "output_field": "count" if aggregation == "count" else f"{aggregation}_{measure}" if measure else aggregation,
        "bar_mode": bar_mode,
        "show_legend": True if series_field else None,
        "show_data_labels": bool(re.search(r"\bdata labels?\b|\bvalue labels?\b|\blabels?\b", desc_lower)) or None,
        "title": None,
        "x_axis_title": x_field.replace("_", " ").title() if x_field else None,
        "y_axis_title": "Count" if aggregation == "count" else aggregation.title(),
    }


def _parse_chart_requests(prompt: str) -> list[dict[str, Any]]:
    """Parse multiple chart requests from a prompt.

    Splits the prompt on connectors (and also, also, next, finally, then)
    and detects a chart type in each segment.
    """
    charts: list[dict[str, Any]] = []
    seen_descriptions: set[str] = set()
    matches = list(_CHART_HEAD_PATTERN.finditer(prompt))

    for index, match in enumerate(matches):
        chart_type_raw = match.group("type").strip().lower()
        desc_start = match.end()
        desc_end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        description = prompt[desc_start:desc_end].strip(" ,.;:-")
        description = re.sub(
            r"^(?:showing|displaying|of|for|that|comparing|with|having)\s+",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip()
        description = re.sub(
            r"\s+(?:and|then|also)\s*$",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip()

        if not description:
            continue

        desc_key = description.lower()[:80]
        if desc_key in seen_descriptions:
            continue
        seen_descriptions.add(desc_key)

        charts.append(_extract_visual_semantics(description, chart_type_raw))

    return charts


def enrich_intent_with_visualization(
    canonical_intent: dict[str, Any],
) -> dict[str, Any]:
    """Enrich a canonical intent dict with VisualizeIntent actions for each detected chart.

    Supports multi-chart prompts by parsing individual chart requests.
    Falls back to single-chart detection via TriggerDetector for simpler prompts.

    Args:
        canonical_intent: A canonical intent dictionary.

    Returns:
        The same dict with VisualizeIntent actions appended.

    Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
    """
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    prompt = canonical_intent.get("original_prompt", "")
    if not prompt or not isinstance(prompt, str):
        return canonical_intent

    actions = canonical_intent.get("actions", [])
    if not isinstance(actions, list):
        actions = []
        canonical_intent["actions"] = actions

    # Skip if visualize actions already exist AND match the detected count
    existing_viz_count = sum(1 for a in actions if isinstance(a, dict) and a.get("kind") == "visualize")
    if existing_viz_count > 0:
        # If we detect more charts than exist, replace with our detection
        chart_requests = _parse_chart_requests(prompt)
        if len(chart_requests) > existing_viz_count:
            # Remove existing visualize actions and add all detected ones
            actions[:] = [a for a in actions if not (isinstance(a, dict) and a.get("kind") == "visualize")]
            for chart in chart_requests:
                actions.append(chart)
        return canonical_intent

    # Try multi-chart parsing first
    chart_requests = _parse_chart_requests(prompt)

    if chart_requests:
        prompt_lower = prompt.lower()
        for chart in chart_requests:
            if chart.get("show_data_labels") is None and re.search(r"\bdata labels?\b|\bvalue labels?\b", prompt_lower):
                chart["show_data_labels"] = True
            if chart.get("show_legend") is None and "legend" in prompt_lower:
                chart["show_legend"] = True
            actions.append(chart)
        return canonical_intent

    # Fallback: single-chart detection via TriggerDetector
    result = _detector.detect(prompt)
    if not result.triggered:
        return canonical_intent

    actions.append({
        "kind": "visualize",
        "chart_type": result.chart_type_hint,
        "fields": [],
        "group_by": None,
        "aggregation": "count",
    })

    return canonical_intent


def should_produce_visualization(prompt: str) -> bool:
    """Check whether a prompt should produce a visualization intent."""
    if not prompt or not isinstance(prompt, str):
        return False
    # Check multi-chart
    if _parse_chart_requests(prompt):
        return True
    # Fallback single-chart
    result = _detector.detect(prompt)
    return result.triggered


__all__ = [
    "enrich_intent_with_visualization",
    "should_produce_visualization",
]
