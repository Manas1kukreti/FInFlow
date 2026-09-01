"""Unit tests for the TriggerDetector visualization trigger language detection."""

import pytest

from finflow_agent.planning.trigger_detector import (
    ANALYTICAL_ONLY_TERMS,
    TRIGGER_PHRASES,
    TRIGGER_TERMS,
    TriggerDetector,
    TriggerResult,
)


@pytest.fixture
def detector() -> TriggerDetector:
    return TriggerDetector()


class TestTriggerTermsDefinition:
    """Verify correct definition of trigger term sets."""

    def test_trigger_terms_contains_expected_words(self):
        expected = {"chart", "graph", "plot", "visualize", "visualise", "visualization", "visualisation"}
        assert TRIGGER_TERMS == expected

    def test_trigger_phrases_contains_expected_phrases(self):
        expected = {"pie chart", "bar chart", "line chart", "scatter plot", "histogram", "as a chart", "as a graph"}
        assert set(TRIGGER_PHRASES) == expected

    def test_analytical_only_terms_contains_expected_words(self):
        expected = {"trend", "distribution", "compare", "breakdown", "summary", "overview", "analysis"}
        assert ANALYTICAL_ONLY_TERMS == expected


class TestBasicTriggerDetection:
    """Test that trigger terms activate visualization."""

    @pytest.mark.parametrize("term", sorted(TRIGGER_TERMS))
    def test_single_trigger_term_activates(self, detector: TriggerDetector, term: str):
        result = detector.detect(f"show me a {term} of the data")
        assert result.triggered is True
        assert result.matched_term == term

    @pytest.mark.parametrize("term", sorted(TRIGGER_TERMS))
    def test_case_insensitive_upper(self, detector: TriggerDetector, term: str):
        result = detector.detect(f"show me a {term.upper()} of the data")
        assert result.triggered is True

    @pytest.mark.parametrize("term", sorted(TRIGGER_TERMS))
    def test_case_insensitive_mixed(self, detector: TriggerDetector, term: str):
        mixed = term[0].upper() + term[1:]
        result = detector.detect(f"show me a {mixed} of the data")
        assert result.triggered is True


class TestPhraseDetection:
    """Test multi-word phrase detection with chart type hints."""

    @pytest.mark.parametrize("phrase,hint", [
        ("pie chart", "pie"),
        ("bar chart", "bar"),
        ("line chart", "line"),
        ("scatter plot", "scatter"),
        ("histogram", "histogram"),
    ])
    def test_phrase_returns_chart_type_hint(self, detector: TriggerDetector, phrase: str, hint: str):
        result = detector.detect(f"show me a {phrase} of revenue")
        assert result.triggered is True
        assert result.matched_term == phrase
        assert result.chart_type_hint == hint

    def test_as_a_chart_phrase(self, detector: TriggerDetector):
        result = detector.detect("display revenue as a chart")
        assert result.triggered is True
        assert result.matched_term == "as a chart"
        assert result.chart_type_hint is None

    def test_as_a_graph_phrase(self, detector: TriggerDetector):
        result = detector.detect("display revenue as a graph")
        assert result.triggered is True
        assert result.matched_term == "as a graph"
        assert result.chart_type_hint is None

    def test_phrase_case_insensitive(self, detector: TriggerDetector):
        result = detector.detect("show me a PIE CHART")
        assert result.triggered is True
        assert result.chart_type_hint == "pie"

    def test_non_contiguous_phrase_words_do_not_match_as_phrase(self, detector: TriggerDetector):
        # "pie" and "chart" not contiguous — should match "chart" as single term
        result = detector.detect("I want a pie and also a chart")
        assert result.triggered is True
        assert result.matched_term == "chart"
        assert result.chart_type_hint is None


class TestWholeWordBoundary:
    """Test that substrings within larger words do NOT trigger."""

    @pytest.mark.parametrize("word", [
        "uncharted",
        "recharted",
        "chartreuse",
    ])
    def test_chart_substring_does_not_trigger(self, detector: TriggerDetector, word: str):
        result = detector.detect(f"The {word} territory was vast")
        assert result.triggered is False

    @pytest.mark.parametrize("word", [
        "graphite",
        "paragraph",
        "autograph",
        "graphene",
    ])
    def test_graph_substring_does_not_trigger(self, detector: TriggerDetector, word: str):
        result = detector.detect(f"The {word} was interesting")
        assert result.triggered is False

    @pytest.mark.parametrize("word", [
        "plotter",
        "plotline",
        "subplot",
        "complot",
    ])
    def test_plot_substring_does_not_trigger(self, detector: TriggerDetector, word: str):
        result = detector.detect(f"The {word} was complex")
        assert result.triggered is False

    def test_visualizer_does_not_trigger(self, detector: TriggerDetector):
        result = detector.detect("The visualizer tool was broken")
        assert result.triggered is False

    def test_visualized_does_not_trigger(self, detector: TriggerDetector):
        result = detector.detect("They visualized the outcome in their mind")
        assert result.triggered is False


class TestAnalyticalOnlyTerms:
    """Test that analytical-only terms do NOT trigger visualization."""

    @pytest.mark.parametrize("term", sorted(ANALYTICAL_ONLY_TERMS))
    def test_single_analytical_term_does_not_trigger(self, detector: TriggerDetector, term: str):
        result = detector.detect(f"show me the {term} of data")
        assert result.triggered is False

    def test_multiple_analytical_terms_do_not_trigger(self, detector: TriggerDetector):
        result = detector.detect("show me the trend and distribution with a comparison breakdown")
        assert result.triggered is False

    def test_all_analytical_terms_combined(self, detector: TriggerDetector):
        prompt = " ".join(ANALYTICAL_ONLY_TERMS)
        result = detector.detect(prompt)
        assert result.triggered is False


class TestTriggerPrecedence:
    """Test that trigger language takes precedence over analytical terms."""

    def test_trigger_with_analytical_terms(self, detector: TriggerDetector):
        result = detector.detect("show me a chart of the trend distribution")
        assert result.triggered is True

    def test_phrase_trigger_with_analytical_terms(self, detector: TriggerDetector):
        result = detector.detect("create a bar chart showing the trend overview")
        assert result.triggered is True
        assert result.chart_type_hint == "bar"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, detector: TriggerDetector):
        result = detector.detect("")
        assert result.triggered is False
        assert result.matched_term is None
        assert result.chart_type_hint is None

    def test_whitespace_only(self, detector: TriggerDetector):
        result = detector.detect("   \n\t  ")
        assert result.triggered is False

    def test_no_trigger_terms_present(self, detector: TriggerDetector):
        result = detector.detect("calculate the total revenue for Q1")
        assert result.triggered is False

    def test_trigger_term_at_start_of_prompt(self, detector: TriggerDetector):
        result = detector.detect("chart the revenue by month")
        assert result.triggered is True

    def test_trigger_term_at_end_of_prompt(self, detector: TriggerDetector):
        result = detector.detect("show revenue by month as a chart")
        assert result.triggered is True

    def test_trigger_result_dataclass(self):
        result = TriggerResult(triggered=True, matched_term="chart", chart_type_hint=None)
        assert result.triggered is True
        assert result.matched_term == "chart"
        assert result.chart_type_hint is None
