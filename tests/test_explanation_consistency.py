"""
test_explanation_consistency.py
===============================
Regression tests for P4 (explanation correctness).

These tests pin down four invariants that the explainers must satisfy, for the
*served* model artefacts:

1. **Probability mapping** - the displayed `score` is
   `predict_proba(...)[0][positive_class]` and `label == int(score >= threshold)`.
2. **Class alignment** - SHAP contributions explain the positive class
   (1 = phishing / smishing), resolved from the estimator's `classes_`.
3. **Additivity** - `base_value + sum(shap_values)` equals the model output the
   explanation claims to describe (`explained_output`). For the calibrated
   message model this is the *calibrated probability*, not the inner SVM's
   decision function.
4. **Feature-value display** - `feature_value` is the feature's real value for
   the input (URL feature count, TF-IDF weight), never a SHAP value, and no
   payload field is named `value`.

They skip cleanly when the corresponding model has not been trained.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

MODELS_DIR = pathlib.Path("models")
MESSAGE_MODEL = MODELS_DIR / "message_model.joblib"
URL_MODEL = MODELS_DIR / "url_model.joblib"

has_message_model = MESSAGE_MODEL.exists()
has_url_model = URL_MODEL.exists()

HAM_MESSAGE = "Hi mum, I'll be home around 7. Do you need anything from the shop?"
SMISHING_MESSAGE = (
    "URGENT! Your bank account is suspended. Verify now at "
    "http://hsbc-secure-verify-account.xyz/login to avoid closure"
)
PHISHING_URL = "http://paypal-secure-login.verify-account.xyz/update.php?id=123&token=abc"
OTHER_URL = "https://en.wikipedia.org/wiki/France"

ADDITIVITY_TOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the shared alignment helpers (model-independent)
# ─────────────────────────────────────────────────────────────────────────────

def test_positive_class_index_follows_classes_not_a_hardcoded_1():
    """The positive-class column must come from `classes_`, not from index 1."""
    from fraud_phishing.explain.shap_utils import positive_class_index

    class _Clf:
        def __init__(self, classes):
            self.classes_ = np.asarray(classes)

    assert positive_class_index(_Clf([0, 1])) == 1
    # Reversed storage order: the positive class now lives in column 0.
    assert positive_class_index(_Clf([1, 0])) == 0
    # Multi-class without the positive label: fall back to the last column.
    assert positive_class_index(_Clf([0, 2])) == 1


def test_select_class_shap_values_handles_every_supported_shape():
    """Every shape SHAP may return for binary classification must work."""
    from fraud_phishing.explain.shap_utils import select_class_shap_values

    n_features = 3
    expected = np.array([1.0, 2.0, 3.0])

    # list of per-class arrays
    as_list = [np.zeros((1, n_features)), expected.reshape(1, -1)]
    assert np.allclose(select_class_shap_values(as_list, 1), expected)

    # (n_samples, n_features, n_classes)
    as_3d = np.stack([np.zeros((1, n_features)), expected.reshape(1, -1)], axis=2)
    assert as_3d.shape == (1, n_features, 2)
    assert np.allclose(select_class_shap_values(as_3d, 1), expected)

    # (n_samples, n_features) - already the positive-class output
    assert np.allclose(select_class_shap_values(expected.reshape(1, -1), 1), expected)

    # 1-D
    assert np.allclose(select_class_shap_values(expected, 1), expected)


def test_additivity_gap_is_zero_for_an_exact_explanation():
    from fraud_phishing.explain.shap_utils import additivity_gap

    shap_values = np.array([0.25, 0.5, -0.75])
    assert additivity_gap(0.5, shap_values, 0.5) == pytest.approx(0.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Micro-models: class alignment and units for both URL explainer branches
#
# The served URL model currently predicts "phishing" for essentially every
# input, so it cannot exercise both classes end-to-end. These deterministic
# micro-models cover the explainer's class handling independently of the served
# artefact's quality.
# ─────────────────────────────────────────────────────────────────────────────

_X_MICRO = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.9, 1.0],
        [1.0, 0.9],
        [0.8, 1.0],
    ]
)
_Y_MICRO = np.array([0, 0, 0, 1, 1, 1])


def test_url_explainer_explains_tree_model_in_probability_units():
    from sklearn.ensemble import RandomForestClassifier

    from fraud_phishing.explain.url_explainer import _compute_shap_values

    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(_X_MICRO, _Y_MICRO)
    assert list(clf.classes_) == [0, 1]

    result = _compute_shap_values(clf, _X_MICRO)

    assert result["explanation_type"] == "TreeExplainer"
    assert result["units"] == "probability"
    assert result["explained_output"] == "P(phishing)"

    model_output = float(clf.predict_proba(_X_MICRO)[0][1])
    reconstructed = result["base_value"] + float(np.sum(result["shap_values"]))
    assert reconstructed == pytest.approx(model_output, abs=ADDITIVITY_TOL)
    assert result["additivity_gap"] <= ADDITIVITY_TOL
    assert result["shap_values"].shape == (2,)

    # Class alignment: the reconstruction matches P(phishing) for this row and,
    # when the two classes are clearly separated, demonstrably not P(legitimate).
    complement = float(clf.predict_proba(_X_MICRO)[0][0])
    if abs(model_output - complement) > 0.2:
        assert abs(reconstructed - complement) > 0.2


def test_url_explainer_explains_linear_model_in_log_odds_units():
    from sklearn.linear_model import LogisticRegression

    from fraud_phishing.explain.url_explainer import _compute_shap_values

    clf = LogisticRegression(max_iter=1000).fit(_X_MICRO, _Y_MICRO)
    result = _compute_shap_values(clf, _X_MICRO)

    assert result["explanation_type"] == "LinearExplainer"
    assert result["units"] == "log_odds"
    assert result["explained_output"] == "log-odds of P(phishing)"

    # Additive to the decision function (log-odds), which maps monotonically onto
    # the displayed probability via the sigmoid.
    decision = float(clf.decision_function(_X_MICRO)[0])
    reconstructed = result["base_value"] + float(np.sum(result["shap_values"]))
    assert reconstructed == pytest.approx(decision, abs=ADDITIVITY_TOL)
    assert result["additivity_gap"] <= ADDITIVITY_TOL

    # The explained score agrees with the displayed probability through sigmoid.
    probability = float(clf.predict_proba(_X_MICRO)[0][1])
    assert 1.0 / (1.0 + np.exp(-decision)) == pytest.approx(probability, abs=1e-9)


def test_url_explainer_direction_follows_positive_class():
    """A feature lowering P(phishing) must get a negative contribution."""
    from sklearn.linear_model import LogisticRegression

    from fraud_phishing.explain.url_explainer import _compute_shap_values

    # Row 0 is the positive class; both features correlate with it.
    X = np.array([[3.0, 3.0], [-3.0, -3.0]])
    y = np.array([1, 0])
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    assert float(clf.predict_proba(X)[0][1]) > 0.5

    positive_row = _compute_shap_values(clf, X)
    assert np.all(positive_row["shap_values"] > 0)

    # Same model, explaining a row the model scores as legitimate: both
    # contributions must flip sign.
    negative_row = _compute_shap_values(clf, X[::-1])
    assert float(clf.predict_proba(X[::-1])[0][1]) < 0.5
    assert np.all(negative_row["shap_values"] < 0)


# ─────────────────────────────────────────────────────────────────────────────
# Message explainer
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def message_bundle():
    if not has_message_model:
        pytest.skip("Message model not trained.")
    from fraud_phishing.models.message_model import load_model

    pipeline, metadata = load_model()
    return pipeline, metadata


@pytest.mark.skipif(not has_message_model, reason="Message model not trained.")
@pytest.mark.parametrize(
    "text,expected_label",
    [(HAM_MESSAGE, 0), (SMISHING_MESSAGE, 1)],
    ids=["ham", "smishing"],
)
def test_message_score_and_label_mapping(text, expected_label, message_bundle):
    """`score` is P(smishing) and `label` is that score against the threshold."""
    from fraud_phishing.explain.message_explainer import explain_message

    pipeline, metadata = message_bundle
    result = explain_message(text, pipeline=pipeline, metadata=metadata)

    expected_score = float(pipeline.predict_proba([text])[0][1])
    assert result["score"] == pytest.approx(expected_score, abs=1e-12)
    assert result["label"] == expected_label
    assert result["label"] == int(expected_score >= result["threshold"])
    assert result["label_name"] == ("smishing" if expected_label == 1 else "ham")


@pytest.mark.skipif(not has_message_model, reason="Message model not trained.")
@pytest.mark.parametrize(
    "text", [HAM_MESSAGE, SMISHING_MESSAGE], ids=["ham", "smishing"]
)
def test_message_shap_adds_up_to_the_displayed_probability(text, message_bundle):
    """base + sum(all SHAP values) == P(smishing) - the single planted invariant."""
    from fraud_phishing.explain.message_explainer import (
        _build_transform_fn,
        _compute_shap_values,
        explain_message,
    )

    pipeline, metadata = message_bundle
    result = explain_message(text, pipeline=pipeline, metadata=metadata)

    assert result["units"] == "probability"
    assert result["explained_output"] == "P(smishing)"
    assert result["additivity_gap"] is not None
    assert result["additivity_gap"] <= ADDITIVITY_TOL

    # Recompute the full contribution vector (the payload is truncated to top-K).
    transform_fn = _build_transform_fn(pipeline)
    X_instance = transform_fn([text])
    shap_values, explanation_type, base_value, gap = _compute_shap_values(
        pipeline.named_steps["clf"], X_instance, transform_fn, None
    )

    assert explanation_type == "KernelExplainer"
    assert base_value == pytest.approx(result["base_value"], abs=1e-12)
    assert gap <= ADDITIVITY_TOL

    probability = float(pipeline.predict_proba([text])[0][1])
    assert base_value + float(np.sum(shap_values)) == pytest.approx(
        probability, abs=ADDITIVITY_TOL
    )


@pytest.mark.skipif(not has_message_model, reason="Message model not trained.")
def test_message_explanation_does_not_explain_the_uncalibrated_svm(message_bundle):
    """
    Regression for the P4 defect: the served classifier is a
    `CalibratedClassifierCV(LinearSVC)`. The explanation must be anchored to the
    calibrated probability, not to the inner SVM's decision function.
    """
    from fraud_phishing.explain.message_explainer import (
        _build_transform_fn,
        _compute_shap_values,
    )

    pipeline, _ = message_bundle
    clf = pipeline.named_steps["clf"]
    assert hasattr(clf, "calibrated_classifiers_"), "Expected a calibrated classifier"

    transform_fn = _build_transform_fn(pipeline)
    X_instance = transform_fn([SMISHING_MESSAGE])
    shap_values, _, base_value, _ = _compute_shap_values(
        clf, X_instance, transform_fn, None
    )

    inner = next(
        (
            getattr(cal, "estimator", None)
            for cal in clf.calibrated_classifiers_
            if getattr(getattr(cal, "estimator", None), "coef_", None) is not None
        ),
        None,
    )
    assert inner is not None, "Could not locate the inner linear estimator"

    reconstructed = base_value + float(np.sum(shap_values))
    probability = float(pipeline.predict_proba([SMISHING_MESSAGE])[0][1])
    decision = float(inner.decision_function(X_instance)[0])

    assert reconstructed == pytest.approx(probability, abs=ADDITIVITY_TOL)
    # ...and is demonstrably *not* the uncalibrated decision function.
    assert abs(reconstructed - decision) > 0.05


@pytest.mark.skipif(not has_message_model, reason="Message model not trained.")
@pytest.mark.parametrize(
    "text", [HAM_MESSAGE, SMISHING_MESSAGE], ids=["ham", "smishing"]
)
def test_message_direction_agrees_with_shap_sign(text, message_bundle):
    from fraud_phishing.explain.message_explainer import explain_message

    pipeline, metadata = message_bundle
    result = explain_message(text, pipeline=pipeline, metadata=metadata)

    contributions = result["top_positive_features"] + result["top_negative_features"]
    assert contributions, "Expected at least one SHAP contribution"
    for item in result["top_positive_features"]:
        assert item["shap_value"] > 0
        assert item["direction"] == "increases_risk"
    for item in result["top_negative_features"]:
        assert item["shap_value"] < 0
        assert item["direction"] == "decreases_risk"


@pytest.mark.skipif(not has_message_model, reason="Message model not trained.")
def test_message_feature_value_is_the_tfidf_weight_not_the_shap_value(message_bundle):
    """`feature_value` must be the real TF-IDF weight of that token."""
    from fraud_phishing.explain.message_explainer import (
        _build_transform_fn,
        _get_feature_names,
        explain_message,
    )

    pipeline, metadata = message_bundle
    result = explain_message(SMISHING_MESSAGE, pipeline=pipeline, metadata=metadata)

    columns = _get_feature_names(pipeline)
    names = [display_name for display_name, _, _ in columns]
    assert len(names) == len(set(names)), "Display names must be unique"

    row = _build_transform_fn(pipeline)([SMISHING_MESSAGE]).tocoo()
    weights = {int(j): float(v) for j, v in zip(row.col, row.data)}

    checked = 0
    for item in result["top_positive_features"] + result["top_negative_features"]:
        index = names.index(item["feature"])
        expected = weights.get(index, 0.0)
        assert item["feature_value"] == pytest.approx(expected, abs=1e-12)
        assert "value" not in item, "Ambiguous 'value' field must not be emitted"
        checked += 1

    assert checked > 0
    # With an empty-message SHAP baseline every reported token is present in the
    # input, so the feature values are genuinely non-zero.
    assert any(item["feature_value"] > 0 for item in result["top_positive_features"])


# ─────────────────────────────────────────────────────────────────────────────
# URL explainer against the served artefact
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def url_bundle():
    if not has_url_model:
        pytest.skip("URL model not trained.")
    from fraud_phishing.models.url_model import load_model

    pipeline, metadata = load_model()
    return pipeline, metadata


@pytest.mark.skipif(not has_url_model, reason="URL model not trained.")
@pytest.mark.parametrize("url", [PHISHING_URL, OTHER_URL, "www.example.com/path?q=1"])
def test_url_score_and_label_mapping(url, url_bundle):
    from fraud_phishing.features.url_features import extract_features

    from fraud_phishing.explain.url_explainer import explain_url

    pipeline, metadata = url_bundle
    result = explain_url(url, pipeline=pipeline, metadata=metadata)

    feature_names = list(metadata["feature_names"])
    features = extract_features(url)
    X = np.array([[features[name] for name in feature_names]], dtype=float)
    expected_score = float(pipeline.predict_proba(X)[0][1])

    assert result["score"] == pytest.approx(expected_score, abs=1e-12)
    assert result["label"] == int(expected_score >= result["threshold"])
    assert result["label_name"] == ("phishing" if result["label"] == 1 else "legitimate")


@pytest.mark.skipif(not has_url_model, reason="URL model not trained.")
@pytest.mark.parametrize("url", [PHISHING_URL, OTHER_URL, "www.example.com/path?q=1"])
def test_url_shap_adds_up_to_the_explained_output(url, url_bundle):
    """Additivity pins the explainer to the class and units it advertises."""
    from fraud_phishing.features.url_features import extract_features

    from fraud_phishing.explain.url_explainer import _compute_shap_values, explain_url

    pipeline, metadata = url_bundle
    result = explain_url(url, pipeline=pipeline, metadata=metadata)

    feature_names = list(metadata["feature_names"])
    features = extract_features(url)
    X = np.array([[features[name] for name in feature_names]], dtype=float)
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    clf = pipeline.named_steps["clf"]

    internal = _compute_shap_values(clf, X_scaled)
    reconstructed = internal["base_value"] + float(np.sum(internal["shap_values"]))

    assert result["units"] == internal["units"]
    assert result["explained_output"] == internal["explained_output"]
    assert result["base_value"] == pytest.approx(internal["base_value"], abs=1e-12)

    if internal["units"] == "probability":
        assert reconstructed == pytest.approx(result["score"], abs=ADDITIVITY_TOL)
    else:
        decision = float(clf.decision_function(X_scaled)[0])
        assert reconstructed == pytest.approx(decision, abs=ADDITIVITY_TOL)

    assert internal["additivity_gap"] <= ADDITIVITY_TOL


@pytest.mark.skipif(not has_url_model, reason="URL model not trained.")
def test_url_feature_value_matches_the_feature_extractor(url_bundle):
    from fraud_phishing.features.url_features import extract_features

    from fraud_phishing.explain.url_explainer import explain_url

    pipeline, metadata = url_bundle
    result = explain_url(PHISHING_URL, pipeline=pipeline, metadata=metadata)
    features = extract_features(PHISHING_URL)

    contributions = result["top_positive_features"] + result["top_negative_features"]
    assert contributions, "Expected at least one contribution"
    for item in contributions:
        assert item["feature_value"] == pytest.approx(
            float(features[item["feature"]]), abs=1e-12
        )
        assert "value" not in item
        assert item["description"], "Every feature must carry a human-readable description"


@pytest.mark.skipif(not has_url_model, reason="URL model not trained.")
def test_url_direction_agrees_with_shap_sign(url_bundle):
    from fraud_phishing.explain.url_explainer import explain_url

    pipeline, metadata = url_bundle
    result = explain_url(PHISHING_URL, pipeline=pipeline, metadata=metadata)

    assert result["top_positive_features"], "Expected at least one risk factor"
    for item in result["top_positive_features"]:
        assert item["shap_value"] > 0
        assert item["direction"] == "increases_risk"
    for item in result["top_negative_features"]:
        assert item["shap_value"] < 0
        assert item["direction"] == "decreases_risk"


# ─────────────────────────────────────────────────────────────────────────────
# API payload
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_client():
    """TestClient with the real model registry populated (skips if untrained)."""
    from fastapi.testclient import TestClient

    from fraud_phishing.api.main import app
    from fraud_phishing.api.services import registry

    registry.load_models()
    if not registry.message_loaded:
        pytest.skip("Message model not loaded.")
    return TestClient(app, raise_server_exceptions=False)


def test_api_message_explanation_payload_is_unambiguous(loaded_client):
    """No field named `value`; SHAP and feature values are separate fields."""
    resp = loaded_client.post("/api/analyze/message", json={"message": SMISHING_MESSAGE})
    assert resp.status_code == 200
    explanation = resp.json()["explanation"]
    assert explanation is not None

    assert explanation["units"] == "probability"
    assert explanation["explained_output"] == "P(smishing)"
    assert explanation["base_value"] is not None
    assert explanation["additivity_gap"] <= ADDITIVITY_TOL

    items = explanation["top_positive_features"] + explanation["top_negative_features"]
    assert items
    for item in items:
        assert "value" not in item
        assert set(item) >= {"feature", "shap_value", "feature_value", "direction"}
        assert isinstance(item["shap_value"], (int, float))

    # The displayed score is the message probability, and the label agrees.
    body = resp.json()
    assert body["message_label"] == int(
        body["message_score"] >= body["message_threshold"]
    )


def test_api_url_explanation_payload_is_unambiguous(loaded_client):
    resp = loaded_client.post("/api/analyze/url", json={"url": PHISHING_URL})
    assert resp.status_code == 200
    explanation = resp.json()["explanation"]
    assert explanation is not None

    assert explanation["units"] in {"probability", "log_odds"}
    assert explanation["explained_output"]
    assert explanation["additivity_gap"] is not None

    items = explanation["top_positive_features"] + explanation["top_negative_features"]
    assert items
    for item in items:
        assert "value" not in item
        assert set(item) >= {"feature", "shap_value", "feature_value", "direction"}
        assert item["description"]
