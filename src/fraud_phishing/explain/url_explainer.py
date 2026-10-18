"""
url_explainer.py
================
SHAP-based explanations for the URL phishing classifier.

Uses TreeExplainer for Random Forest (exact, fast) or
LinearExplainer for Logistic Regression.
"""

import pathlib
import sys
from typing import Any

import numpy as np
import shap

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from fraud_phishing.features.url_features import FEATURE_NAMES, extract_features
from fraud_phishing.models.url_model import load_model

_TOP_K = 15

# Human-readable descriptions for each feature
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "url_length": "Total URL length in characters",
    "hostname_length": "Length of the hostname",
    "path_length": "Length of the URL path",
    "query_length": "Length of the query string",
    "registered_domain_length": "Length of the registered domain (e.g. example.com)",
    "dot_count": "Number of dots in the URL",
    "hyphen_count": "Number of hyphens in the URL",
    "slash_count": "Number of forward slashes",
    "underscore_count": "Number of underscores",
    "at_count": "Number of @ symbols (often used to obscure host)",
    "question_count": "Number of question marks",
    "equals_count": "Number of equals signs (query parameters)",
    "ampersand_count": "Number of & signs (query parameters)",
    "percent_count": "Number of % signs (URL encoding)",
    "digit_count": "Number of digits in the URL",
    "digit_ratio": "Fraction of digits in the URL",
    "special_char_ratio": "Fraction of special characters",
    "tilde_count": "Number of tilde (~) characters",
    "subdomain_count": "Number of subdomain levels",
    "subdomain_length": "Total length of subdomain string",
    "tld_length": "Length of the top-level domain suffix",
    "path_depth": "Number of path segments (directory depth)",
    "query_param_count": "Number of distinct query parameters",
    "has_query": "Whether the URL has a query string (0/1)",
    "is_https": "Whether the URL uses HTTPS (0/1)",
    "is_ip_host": "Whether the host is an IP address (0/1)",
    "has_at_symbol": "Whether URL contains an @ symbol (0/1)",
    "has_percent_encoding": "Whether URL contains percent-encoded characters (0/1)",
    "is_url_shortener": "Whether the domain is a known URL shortener (0/1)",
    "suspicious_token_count": "Count of documented suspicious keyword tokens in URL",
}


def explain_url(
    url: str,
    pipeline=None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Generate a SHAP explanation for a single URL.

    Parameters
    ----------
    url : str
        Raw URL string.
    pipeline : sklearn.pipeline.Pipeline or None
        Fitted URL pipeline. If None, loads from models/.
    metadata : dict or None
        Model metadata. If None, loads from models/.

    Returns
    -------
    dict with keys:
        model_name, url, label, label_name, score, threshold,
        features, top_positive_features, top_negative_features,
        disclaimer
    """
    if pipeline is None:
        pipeline, metadata = load_model()

    metadata = metadata or {}
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")

    # Extract features
    try:
        feat_dict = extract_features(url)
    except ValueError as exc:
        return {
            "error": str(exc),
            "url": url,
            "label": None,
            "disclaimer": _DISCLAIMER,
        }

    X_instance = np.array([[feat_dict[name] for name in FEATURE_NAMES]], dtype=float)

    # Apply scaler from pipeline
    try:
        scaler = pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X_instance)
    except Exception:
        X_scaled = X_instance

    clf = pipeline.named_steps["clf"]

    # Prediction
    proba = pipeline.predict_proba(X_instance)[0][1]
    label = int(proba >= threshold)

    # SHAP
    try:
        shap_values, explanation_type = _compute_shap_values(clf, X_scaled)
    except Exception as exc:
        return {
            "model_name": model_name,
            "url": url,
            "label": label,
            "label_name": "phishing" if label == 1 else "legitimate",
            "score": float(proba),
            "threshold": threshold,
            "features": feat_dict,
            "top_positive_features": [],
            "top_negative_features": [],
            "explanation_type": "unavailable",
            "explanation_error": str(exc),
            "disclaimer": _DISCLAIMER,
        }

    contributions = list(zip(FEATURE_NAMES, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_positive = [
        {
            "feature": f,
            "value": feat_dict[f],
            "shap_value": float(v),
            "direction": "increases_risk",
            "description": FEATURE_DESCRIPTIONS.get(f, f),
        }
        for f, v in contributions
        if v > 0
    ][:_TOP_K]

    top_negative = [
        {
            "feature": f,
            "value": feat_dict[f],
            "shap_value": float(v),
            "direction": "decreases_risk",
            "description": FEATURE_DESCRIPTIONS.get(f, f),
        }
        for f, v in contributions
        if v < 0
    ][:_TOP_K]

    return {
        "model_name": model_name,
        "url": url,
        "label": label,
        "label_name": "phishing" if label == 1 else "legitimate",
        "score": float(proba),
        "threshold": threshold,
        "features": feat_dict,
        "explanation_type": explanation_type,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "disclaimer": _DISCLAIMER,
    }


def _compute_shap_values(clf, X_scaled: np.ndarray) -> tuple[list[float], str]:
    """
    Compute SHAP values using the most appropriate explainer.
    Tries TreeExplainer → LinearExplainer → KernelExplainer.
    """
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    # Tree-based model
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_scaled)
        # sv is list[class_0_values, class_1_values] for binary classification
        if isinstance(sv, list) and len(sv) == 2:
            return sv[1][0].tolist(), "TreeExplainer"
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            return sv[0, :, 1].tolist(), "TreeExplainer"
        return sv[0].tolist(), "TreeExplainer"

    # Linear model
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, np.zeros_like(X_scaled))
        sv = explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            return sv[1][0].tolist(), "LinearExplainer"
        return sv[0].tolist(), "LinearExplainer"

    # Generic fallback
    bg = np.zeros((10, X_scaled.shape[1]))

    def pred_fn(X):
        return clf.predict_proba(X)

    explainer = shap.KernelExplainer(pred_fn, bg)
    sv = explainer.shap_values(X_scaled, nsamples=100, silent=True)
    if isinstance(sv, list):
        return sv[1][0].tolist(), "KernelExplainer"
    return sv[0].tolist(), "KernelExplainer"


_DISCLAIMER = (
    "This is a model-based risk assessment. SHAP values show which features "
    "influenced the model's prediction, not causal proof that a URL is malicious. "
    "HTTPS, the word 'login', or URL shorteners also appear in legitimate URLs. "
    "Verify important links through official channels."
)
