"""E2E: Streamlit UI pure-Python logic — no browser required.

Tests the stateless helper functions in apps/streamlit_ui/app.py:
feature extraction from text, missing-field detection, metric parsing,
report section splitting, and number highlighting.

These run without launching a Streamlit server or a real LLM.
"""

import math

import pytest

# Import individual helpers (not the Streamlit rendering layer)
from apps.streamlit_ui.app import (
    DIABETES_FEATURE_INFO,
    DIABETES_FEATURES,
    KEY_DIABETES_FEATURES,
    _coerce_optional_float,
    _extract_json_object,
    _extract_metrics,
    _has_enough_diabetes_data,
    _highlight_numbers,
    _is_missing,
    _missing_diabetes_fields,
    _regex_feature_fallback,
    _report_sections,
    _risk_tone,
    _result_text,
    _split_sentences,
)
from packages.schemas.patient import PatientRecord


# ---------------------------------------------------------------------------
# _coerce_optional_float
# ---------------------------------------------------------------------------

class TestCoerceOptionalFloat:
    def test_none_returns_none(self):
        assert _coerce_optional_float(None) is None

    def test_empty_string_returns_none(self):
        assert _coerce_optional_float("") is None

    def test_nan_returns_none(self):
        assert _coerce_optional_float(float("nan")) is None

    def test_valid_float(self):
        assert _coerce_optional_float(3.14) == pytest.approx(3.14)

    def test_valid_string_float(self):
        assert _coerce_optional_float("42.5") == pytest.approx(42.5)

    def test_invalid_string_returns_none(self):
        assert _coerce_optional_float("abc") is None


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_extracts_embedded_json(self):
        text = 'Here is the result: {"age": 30, "plas": 120} done.'
        result = _extract_json_object(text)
        assert result == {"age": 30, "plas": 120}

    def test_returns_empty_dict_when_no_json(self):
        assert _extract_json_object("no json here") == {}

    def test_returns_empty_dict_for_invalid_json(self):
        assert _extract_json_object("{invalid}") == {}

    def test_returns_empty_dict_for_non_dict_json(self):
        assert _extract_json_object("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# _regex_feature_fallback
# ---------------------------------------------------------------------------

class TestRegexFeatureFallback:
    def test_extracts_age_from_english(self):
        result = _regex_feature_fallback("I am 45 years old")
        assert result.get("age") == 45.0

    def test_extracts_glucose(self):
        result = _regex_feature_fallback("blood sugar is 130 mg/dL")
        assert result.get("plas") == 130.0

    def test_extracts_bmi(self):
        result = _regex_feature_fallback("BMI 27.5")
        assert result.get("mass") == 27.5

    def test_extracts_blood_pressure(self):
        result = _regex_feature_fallback("blood pressure 120/80")
        assert result.get("pres") == 80.0

    def test_extracts_insulin(self):
        result = _regex_feature_fallback("insulin level 85")
        assert result.get("insu") == 85.0

    def test_empty_message_returns_empty_dict(self):
        assert _regex_feature_fallback("") == {}

    def test_unrelated_message_returns_empty_dict(self):
        assert _regex_feature_fallback("I feel great today") == {}


# ---------------------------------------------------------------------------
# _is_missing / _missing_diabetes_fields / _has_enough_diabetes_data
# ---------------------------------------------------------------------------

def _make_record(features: dict) -> PatientRecord:
    return PatientRecord(
        handle="test-001",
        age=40,
        sex="female",
        features=features,
        notes="test",
    )


class TestMissingFieldDetection:
    def test_none_is_missing(self):
        assert _is_missing(None)

    def test_nan_is_missing(self):
        assert _is_missing(float("nan"))

    def test_zero_is_not_missing(self):
        assert not _is_missing(0.0)

    def test_missing_diabetes_fields_all_absent(self):
        record = _make_record({})
        missing = _missing_diabetes_fields(record)
        assert set(missing) == set(DIABETES_FEATURES)

    def test_missing_diabetes_fields_some_present(self):
        record = _make_record({"plas": 120.0, "age": 35.0})
        missing = _missing_diabetes_fields(record)
        assert "plas" not in missing
        assert "age" not in missing
        assert len(missing) == len(DIABETES_FEATURES) - 2

    def test_has_enough_data_with_two_key_features(self):
        record = _make_record({"plas": 120.0, "age": 35.0})
        assert _has_enough_diabetes_data(record)

    def test_not_enough_data_with_one_key_feature(self):
        record = _make_record({"plas": 120.0})
        assert not _has_enough_diabetes_data(record)

    def test_not_enough_data_with_no_key_features(self):
        record = _make_record({"preg": 2.0, "skin": 20.0})
        assert not _has_enough_diabetes_data(record)


# ---------------------------------------------------------------------------
# _risk_tone
# ---------------------------------------------------------------------------

class TestRiskTone:
    def test_high_risk(self):
        assert _risk_tone(90.0) == "high"

    def test_medium_risk(self):
        assert _risk_tone(75.0) == "medium"

    def test_low_risk(self):
        assert _risk_tone(50.0) == "low"

    def test_boundary_85_is_high(self):
        assert _risk_tone(85.0) == "high"

    def test_boundary_70_is_medium(self):
        assert _risk_tone(70.0) == "medium"


# ---------------------------------------------------------------------------
# _extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_extracts_breast_cancer_metric(self):
        text = "The breast malignancy probability is 72%."
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Breast Screening" in labels

    def test_extracts_diabetes_metric(self):
        text = "Diabetes risk is estimated at 65%."
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Diabetes Risk" in labels

    def test_extracts_glucose_metric(self):
        text = "Plasma glucose: 148 mg/dL"
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Glucose" in labels

    def test_returns_empty_for_plain_text(self):
        assert _extract_metrics("No medical values here.") == []


# ---------------------------------------------------------------------------
# _report_sections
# ---------------------------------------------------------------------------

class TestReportSections:
    def test_recommendation_goes_to_recommended_followup(self):
        text = "We recommend a follow-up in 3 months. The result looks normal."
        sections = _report_sections(text)
        assert "Recommended Follow-Up" in sections

    def test_clinical_disclaimer_goes_to_clinical_note(self):
        text = "This does not replace a clinical decision. It supports clinical judgement."
        sections = _report_sections(text)
        assert "Clinical Note" in sections

    def test_general_text_goes_to_key_findings(self):
        text = "The patient shows elevated glucose levels."
        sections = _report_sections(text)
        assert "Key Findings" in sections

    def test_empty_sections_not_included(self):
        text = "Normal results observed."
        sections = _report_sections(text)
        assert "Recommended Follow-Up" not in sections
        assert "Clinical Note" not in sections


# ---------------------------------------------------------------------------
# _highlight_numbers
# ---------------------------------------------------------------------------

class TestHighlightNumbers:
    def test_wraps_number_in_span(self):
        result = _highlight_numbers("Value is 42.5%")
        assert "<span class='metric-inline'>42.5%</span>" in result

    def test_escapes_html_entities(self):
        result = _highlight_numbers("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# _result_text — extracts text from varied BeeAI result shapes
# ---------------------------------------------------------------------------

class TestResultText:
    def test_returns_str_for_plain_object(self):
        class FakeResult:
            def __str__(self):
                return "plain text"

        assert _result_text(FakeResult()) == "plain text"

    def test_prefers_output_structured_response(self):
        class Structured:
            response = "structured response"

        class FakeResult:
            output_structured = Structured()

        assert _result_text(FakeResult()) == "structured response"

    def test_falls_back_to_answer_text(self):
        class Answer:
            text = "answer text"

        class FakeResult:
            output_structured = None
            answer = Answer()

        assert _result_text(FakeResult()) == "answer text"


# ---------------------------------------------------------------------------
# DIABETES_FEATURE_INFO completeness
# ---------------------------------------------------------------------------

class TestFeatureInfoCompleteness:
    def test_all_eight_features_documented(self):
        assert set(DIABETES_FEATURE_INFO.keys()) == set(DIABETES_FEATURES)

    def test_all_key_features_are_valid(self):
        assert KEY_DIABETES_FEATURES <= set(DIABETES_FEATURES)

    def test_each_feature_has_label_and_detail(self):
        for feature, info in DIABETES_FEATURE_INFO.items():
            assert "label" in info, f"{feature} missing label"
            assert "detail" in info, f"{feature} missing detail"
