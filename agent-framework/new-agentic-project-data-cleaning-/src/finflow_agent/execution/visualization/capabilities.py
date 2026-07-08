"""Authoritative visualization capability registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisualizationCapability:
    chart_type: str
    supports_series: bool
    supported_bar_modes: set[str] = field(default_factory=set)
    max_grouping_dimensions: int = 1


VISUALIZATION_CAPABILITIES: dict[str, VisualizationCapability] = {
    "bar": VisualizationCapability(
        chart_type="bar",
        supports_series=True,
        supported_bar_modes={"grouped", "stacked"},
        max_grouping_dimensions=2,
    ),
    "line": VisualizationCapability(
        chart_type="line",
        supports_series=False,
        max_grouping_dimensions=1,
    ),
    "pie": VisualizationCapability(
        chart_type="pie",
        supports_series=False,
        max_grouping_dimensions=1,
    ),
    "scatter": VisualizationCapability(
        chart_type="scatter",
        supports_series=False,
        max_grouping_dimensions=1,
    ),
    "histogram": VisualizationCapability(
        chart_type="histogram",
        supports_series=False,
        max_grouping_dimensions=1,
    ),
    "auto": VisualizationCapability(
        chart_type="auto",
        supports_series=False,
        max_grouping_dimensions=1,
    ),
}


def normalize_chart_type(chart_type: str | None, *, bar_mode: str | None = None) -> tuple[str, str | None]:
    """Normalize legacy chart aliases to the canonical chart contract."""
    normalized = (chart_type or "auto").strip().lower()
    normalized_bar_mode = (bar_mode or "").strip().lower() or None
    if normalized == "stacked_bar":
        normalized = "bar"
        normalized_bar_mode = "stacked"
    return normalized, normalized_bar_mode

