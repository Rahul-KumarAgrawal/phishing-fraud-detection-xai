"""
message_explainer.py
====================
SHAP-based explanations for the message (SMS/smishing) classifier.

Uses LinearExplainer for linear models and KernelExplainer as a fallback.
Explanations are computed on-demand for single inputs, not the whole dataset.
"""

import pathlib
import sys
from typing import Any

import numpy as np
import shap

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from fraud_phishing.models.message_model import load_model

# Maximum number of background samples for KernelExplainer
_KERNEL_BG_SAMPLES = 100
# Maximum number of features to include in explanation output
_TOP_K = 15


def _get_feature_names(pipeline) -> list[str]:
    """Extract feature names from the FeatureUnion TF-IDF pipelines."""
    try:
        union = pipeline.named_steps["features"]
        names = []
        for _, transformer in union.transformer_list:
            names.extend(transformer.get_feature_names_out())
        return names
    except Exception:
        return []


def _build_transform_fn(pipeline):
    """Return a callable that applies only the feature step."""
    def _transform(texts):
        return pipeline.named_steps["features"].transform(texts)
    return _transform


def explain_message(
    text: str,
    pipeline=None,
    metadata: dict | None = None,
    background_texts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a SHAP explanation for a single message.

    Parameters
    ----------
    text : str
        Raw input message.
    pipeline : sklearn.pipeline.Pipeline or None
        Fitted pipeline. If None, loads from models/.
    metadata : dict or None
        Model metadata. If None, loads from models/.
    background_texts : list[str] or None
        Background samples for KernelExplainer. If None, uses dummy texts.

    Returns
    -------
    dict with keys:
        model_name, input_text, label, score, threshold,
        top_positive_features, top_negative_features,
        disclaimer
    """
    if pipeline is None:
        pipeline, metadata = load_model()

    metadata = metadata or {}
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")

    # Get prediction
    proba = pipeline.predict_proba([text])[0][1]
    label = int(proba >= threshold)

    feature_names = _get_feature_names(pipeline)
    transform_fn = _build_transform_fn(pipeline)
    clf = pipeline.named_steps["clf"]

    try:
        shap_values, explanation_type = _compute_shap_values(
            text, clf, transform_fn, feature_names, background_texts
        )
    except Exception as exc:
        # Graceful degradation: return prediction without SHAP
        return {
            "model_name": model_name,
            "input_text": text[:200],
            "label": label,
            "label_name": "smishing" if label == 1 else "ham",
            "score": float(proba),
            "threshold": threshold,
            "top_positive_features": [],
            "top_negative_features": [],
            "explanation_type": "unavailable",
            "explanation_error": str(exc),
            "disclaimer": _DISCLAIMER,
        }

    # Rank features by absolute SHAP value
    contributions = list(zip(feature_names, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_positive = [
        {"feature": f, "value": float(v), "direction": "increases_risk"}
        for f, v in contributions
        if v > 0
    ][:_TOP_K]

    top_negative = [
        {"feature": f, "value": float(v), "direction": "decreases_risk"}
        for f, v in contributions
        if v < 0
    ][:_TOP_K]

    return {
        "model_name": model_name,
        "input_text": text[:200],
        "label": label,
        "label_name": "smishing" if label == 1 else "ham",
        "score": float(proba),
        "threshold": threshold,
        "explanation_type": explanation_type,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "disclaimer": _DISCLAIMER,
    }


def _compute_shap_values(text, clf, transform_fn, feature_names, background_texts):
    """
    Attempt LinearExplainer first (supported for LR and CalibratedClassifier
    wrapping a linear model), then fall back to KernelExplainer.
    """
    X_instance = transform_fn([text])

    # Try LinearExplainer (fast, exact for linear models)
    try:
        # For CalibratedClassifierCV we need the underlying linear estimator
        base_clf = _extract_linear_clf(clf)
        if base_clf is not None:
            # Build a background matrix
            bg = _make_sparse_background(X_instance, n=50)
            explainer = shap.LinearExplainer(base_clf, bg, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_instance)
            # shap_values may be list (multi-class) or array
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class 1
            else:
                sv = shap_values[0]
            return sv.tolist() if hasattr(sv, "tolist") else list(sv), "LinearExplainer"
    except Exception:
        pass

    # Fallback: KernelExplainer with small background
    if background_texts is None:
        background_texts = ["ok thanks", "hello", "free prize"]
    bg_X = transform_fn(background_texts[:_KERNEL_BG_SAMPLES])

    def predict_proba_fn(X):
        return clf.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba_fn, bg_X)
    shap_vals = explainer.shap_values(X_instance, nsamples=100, silent=True)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]
    return list(sv), "KernelExplainer"


def _extract_linear_clf(clf):
    """Try to extract a linear classifier from CalibratedClassifierCV or return directly."""
    # CalibratedClassifierCV
    if hasattr(clf, "calibrated_classifiers_"):
        for cal_clf in clf.calibrated_classifiers_:
            base = getattr(cal_clf, "estimator", None) or getattr(cal_clf, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                return base
    # Direct linear model
    if hasattr(clf, "coef_"):
        return clf
    return None


def _make_sparse_background(X_instance, n: int):
    """Create a sparse zero background matrix matching X_instance shape."""
    import scipy.sparse as sp  # noqa: PLC0415
    if sp.issparse(X_instance):
        return sp.csr_matrix((n, X_instance.shape[1]), dtype=float)
    return np.zeros((n, X_instance.shape[1]), dtype=float)


_DISCLAIMER = (
    "This is a model-based risk assessment. SHAP values indicate feature influence "
    "on the model's prediction, not causal proof that a message is fraudulent. "
    "Verify important requests through official channels."
)
