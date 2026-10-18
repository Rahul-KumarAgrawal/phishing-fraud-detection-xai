"""
test_explanations.py
====================
Tests for SHAP explanation output schemas.
These tests verify structure, not correctness of SHAP values
(which depend on trained models).
"""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.features.url_features import FEATURE_NAMES


# ── URL explanation schema tests ───────────────────────────────────────────

def test_url_explainer_imports():
    """url_explainer module must be importable."""
    from fraud_phishing.explain.url_explainer import explain_url  # noqa: F401


def test_message_explainer_imports():
    """message_explainer module must be importable."""
    from fraud_phishing.explain.message_explainer import explain_message  # noqa: F401


def test_url_explainer_no_model_returns_error():
    """explain_url without a trained model raises FileNotFoundError."""
    import pathlib
    import unittest.mock as mock

    models_dir = pathlib.Path("models")
    url_model_path = models_dir / "url_model.joblib"

    if url_model_path.exists():
        pytest.skip("URL model is trained – skip no-model test.")

    from fraud_phishing.explain.url_explainer import explain_url
    with pytest.raises(FileNotFoundError):
        explain_url("http://example.com/login")


def test_message_explainer_no_model_raises():
    """explain_message without a trained model raises FileNotFoundError."""
    msg_model_path = pathlib.Path("models") / "message_model.joblib"
    if msg_model_path.exists():
        pytest.skip("Message model is trained – skip no-model test.")

    from fraud_phishing.explain.message_explainer import explain_message
    with pytest.raises(FileNotFoundError):
        explain_message("Congratulations! You won a prize!")


def test_feature_descriptions_cover_all_features():
    """Every feature name must have a human-readable description."""
    from fraud_phishing.explain.url_explainer import FEATURE_DESCRIPTIONS
    missing = set(FEATURE_NAMES) - set(FEATURE_DESCRIPTIONS.keys())
    assert not missing, f"Features missing descriptions: {missing}"


def test_explanation_output_schema_if_model_exists():
    """If the URL model is trained, verify explanation output schema."""
    url_model_path = pathlib.Path("models") / "url_model.joblib"
    if not url_model_path.exists():
        pytest.skip("URL model not yet trained.")

    from fraud_phishing.explain.url_explainer import explain_url
    result = explain_url("https://example.com/secure/login?id=test")

    required_keys = {"model_name", "url", "label", "label_name", "score", "threshold", "disclaimer"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    assert result["label"] in (0, 1)
    assert result["label_name"] in ("phishing", "legitimate")
    assert 0.0 <= result["score"] <= 1.0

    # top_positive_features must be a list
    assert isinstance(result.get("top_positive_features", []), list)
    assert isinstance(result.get("top_negative_features", []), list)


def test_message_explanation_schema_if_model_exists():
    """If the message model is trained, verify explanation output schema."""
    msg_model_path = pathlib.Path("models") / "message_model.joblib"
    if not msg_model_path.exists():
        pytest.skip("Message model not yet trained.")

    from fraud_phishing.explain.message_explainer import explain_message
    result = explain_message("You have won a free prize! Click here to claim.")

    required_keys = {"model_name", "label", "label_name", "score", "threshold", "disclaimer"}
    assert required_keys.issubset(result.keys())
    assert result["label"] in (0, 1)
    assert 0.0 <= result["score"] <= 1.0
