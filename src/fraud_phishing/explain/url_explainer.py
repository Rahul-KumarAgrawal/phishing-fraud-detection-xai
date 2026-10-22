"""
url_explainer.py
================
SHAP-based explanations for the URL phishing classifier.

Uses TreeExplainer for Random Forest (exact, explains the predicted
probability) or LinearExplainer for Logistic Regression (exact, explains the
log-odds that map monotonically onto that probability).

Every payload states which model output it explains (``explained_output``) and
in which units (``units``), plus ``base_value`` and the ``additivity_gap``
check, so a contribution scale can never be mistaken for the other. The class
column is resolved from the estimator's ``classes_`` (see ``shap_utils``).
"""

import pathlib
import sys
import warnings
from typing import Any

import numpy as np
import shap

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from fraud_phishing.explain.shap_utils import (
    additivity_gap,
    expected_value_for,
    positive_class_index,
    select_class_shap_values,
)
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
    "brand_token_count": "Number of distinct known brand tokens in the hostname",
    "brand_domain_mismatch": (
        "Brand token present but the registered domain is not that brand (0/1)"
    ),
    "brand_lookalike_domain": (
        "Registered domain is a near-miss (edit distance 1-2) of a known brand (0/1)"
    ),
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
        model_name, url, label, label_name, score, threshold, features,
        explanation_type, units, explained_output, base_value,
        additivity_gap, top_positive_features, top_negative_features,
        disclaimer
    """
    if pipeline is None:
        pipeline, metadata = load_model()

    metadata = metadata or {}
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")
    # A model may be trained on a subset of the canonical schema; always use the
    # feature list it was fitted with (in that order).
    feature_names = list(metadata.get("feature_names") or FEATURE_NAMES)

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

    X_instance = np.array([[feat_dict[name] for name in feature_names]], dtype=float)

    # Apply scaler from pipeline
    try:
        scaler = pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X_instance)
    except Exception:
        X_scaled = X_instance

    clf = pipeline.named_steps["clf"]

    # Prediction: score and label both come from the served pipeline.
    proba = float(pipeline.predict_proba(X_instance)[0][1])
    label = int(proba >= threshold)

    # SHAP
    try:
        shap_result = _compute_shap_values(clf, X_scaled)
    except Exception as exc:
        return {
            "model_name": model_name,
            "url": url,
            "label": label,
            "label_name": "phishing" if label == 1 else "legitimate",
            "score": proba,
            "threshold": threshold,
            "features": feat_dict,
            "explanation_type": "unavailable",
            "units": "probability",
            "explained_output": "P(phishing)",
            "base_value": None,
            "additivity_gap": None,
            "top_positive_features": [],
            "top_negative_features": [],
            "explanation_error": str(exc),
            "disclaimer": _DISCLAIMER,
        }

    contributions = sorted(
        zip(feature_names, shap_result["shap_values"]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    def _make(direction: str) -> list[dict[str, Any]]:
        out = []
        for name, value in contributions:
            # "increases risk" means a positive contribution to P(phishing).
            if direction == "increases_risk" and not value > 0:
                continue
            if direction == "decreases_risk" and not value < 0:
                continue
            out.append(
                {
                    "feature": name,
                    "shap_value": float(value),
                    "feature_value": feat_dict[name],
                    "direction": direction,
                    "description": FEATURE_DESCRIPTIONS.get(name, name),
                }
            )
        return out[:_TOP_K]

    return {
        "model_name": model_name,
        "url": url,
        "label": label,
        "label_name": "phishing" if label == 1 else "legitimate",
        "score": proba,
        "threshold": threshold,
        "features": feat_dict,
        "explanation_type": shap_result["explanation_type"],
        "units": shap_result["units"],
        "explained_output": shap_result["explained_output"],
        "base_value": shap_result["base_value"],
        "additivity_gap": shap_result["additivity_gap"],
        "top_positive_features": _make("increases_risk"),
        "top_negative_features": _make("decreases_risk"),
        "disclaimer": _DISCLAIMER,
    }


def _compute_shap_values(clf, X_scaled: np.ndarray) -> dict[str, Any]:
    """
    Compute SHAP values for the positive class with the appropriate explainer.

    Returns a dict with the values, the explainer name, the units of those
    values, the model output they add up to, the base value and the measured
    additivity gap.
    """
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    class_index = positive_class_index(clf)

    def _pack(shap_values, explainer, name, units, explained_output, model_output):
        values = select_class_shap_values(shap_values, class_index)
        base_value = expected_value_for(explainer, class_index)
        return {
            "shap_values": values,
            "explanation_type": name,
            "units": units,
            "explained_output": explained_output,
            "base_value": base_value,
            "additivity_gap": additivity_gap(base_value, values, model_output),
        }

    # Tree-based model: TreeExplainer is exact and explains the class
    # probability, i.e. the same quantity the API returns as `score`.
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_scaled)
        model_output = float(np.asarray(clf.predict_proba(X_scaled))[0][class_index])
        return _pack(raw, explainer, "TreeExplainer", "probability", "P(phishing)", model_output)

    # Linear model: LinearExplainer is exact and explains the log-odds, which
    # map monotonically onto the displayed probability of the same class.
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, np.zeros_like(X_scaled))
        raw = explainer.shap_values(X_scaled)
        decision = np.asarray(clf.decision_function(X_scaled))
        model_output = float(decision[0] if decision.ndim == 1 else decision[0, class_index])
        return _pack(raw, explainer, "LinearExplainer", "log_odds", "log-odds of P(phishing)", model_output)

    # Generic fallback: explain predict_proba directly, in probability units.
    bg = np.zeros((10, X_scaled.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(clf.predict_proba, bg, l1_reg=0)
        raw = explainer.shap_values(X_scaled, nsamples=100, silent=True)
    model_output = float(np.asarray(clf.predict_proba(X_scaled))[0][class_index])
    return _pack(raw, explainer, "KernelExplainer", "probability", "P(phishing)", model_output)



_DISCLAIMER = (
    "This is a model-based risk assessment. SHAP values show which features "
    "influenced the model's prediction, not causal proof that a URL is malicious. "
    "HTTPS, the word 'login', or URL shorteners also appear in legitimate URLs. "
    "Verify important links through official channels."
)
