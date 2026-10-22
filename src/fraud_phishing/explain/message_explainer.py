"""
message_explainer.py
====================
SHAP-based explanations for the message (SMS/smishing) classifier.

The served message classifier is a ``CalibratedClassifierCV``-wrapped
``LinearSVC`` and the API displays its *calibrated* ``predict_proba``. SHAP
therefore explains that same fitted object - the calibrated classifier - via a
model-agnostic ``KernelExplainer``. This guarantees:

* the explained class is the class whose probability is displayed, and
* ``base_value + sum(contributions)`` equals the displayed probability.

An earlier implementation unwrapped the calibration and explained the inner
``LinearSVC`` decision function, so the numbers did not describe the score on
screen. See ``reports/explainability_fixes.md``.

Explanations are computed on demand for a single input, never for a dataset.
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
from fraud_phishing.models.message_model import load_model

# Maximum number of features to include in explanation output
_TOP_K = 15
# Coalitions sampled by KernelExplainer for a single input.
_KERNEL_NSAMPLES = 100

# The default SHAP reference point is an *empty message* (an all-zero TF-IDF
# row). Every reported contribution therefore belongs to a token that is
# actually present in the input, and `base_value` is the model's score for "no
# information". Passing `background_texts` replaces this anchor with a sample of
# reference messages; additivity is unaffected, but contributions then also
# describe tokens that are present in the reference sample and absent from the
# input, which reads confusingly in a UI.
_EMPTY_MESSAGE = ""


def _get_feature_names(pipeline) -> list[tuple[str, str, str]]:
    """
    Return ``(display_name, block, raw_name)`` for every FeatureUnion column.

    Word and character n-grams share a namespace: the word feature ``no`` and
    the character 2-gram ``no`` are different columns with the same raw name, so
    the display name is prefixed with its block (``word__no`` / ``char__no``) to
    keep every reported feature uniquely identifiable in the API and the UI.

    Column order matches ``FeatureUnion.transform`` because both iterate
    ``transformer_list`` in the same order.
    """
    try:
        union = pipeline.named_steps["features"]
        columns: list[tuple[str, str, str]] = []
        for block_name, transformer in union.transformer_list:
            for raw_name in transformer.get_feature_names_out():
                columns.append((f"{block_name}__{raw_name}", block_name, str(raw_name)))
        return columns
    except Exception:
        return []


def _build_transform_fn(pipeline):
    """Return a callable that applies only the feature step."""
    def _transform(texts):
        return pipeline.named_steps["features"].transform(texts)
    return _transform


def _row_values(row) -> dict[int, float]:
    """Map column index -> value for a single-row sparse TF-IDF matrix."""
    coo = row.tocoo()
    return {int(j): float(v) for j, v in zip(coo.col, coo.data)}


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
        Background samples for KernelExplainer. If None, uses a fixed balanced
        reference set (not training data).

    Returns
    -------
    dict with keys:
        model_name, input_text, label, label_name, score, threshold,
        explanation_type, units, explained_output, base_value,
        additivity_gap, top_positive_features, top_negative_features,
        disclaimer
    """
    if pipeline is None:
        pipeline, metadata = load_model()

    metadata = metadata or {}
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")

    # The fitted classifier, wrapper included: this is the object that produces
    # the displayed probability.
    clf = pipeline.named_steps["clf"]

    # Score and label come from the same call the explanation is anchored to.
    proba = float(pipeline.predict_proba([text])[0][1])
    label = int(proba >= threshold)

    feature_columns = _get_feature_names(pipeline)
    transform_fn = _build_transform_fn(pipeline)

    try:
        if not feature_columns:
            raise RuntimeError("Could not resolve TF-IDF feature names from the pipeline")
        X_instance = transform_fn([text])
        shap_values, explanation_type, base_value, gap = _compute_shap_values(
            clf, X_instance, transform_fn, background_texts
        )
    except Exception as exc:
        # Graceful degradation: return prediction without SHAP
        return {
            "model_name": model_name,
            "input_text": text[:200],
            "label": label,
            "label_name": "smishing" if label == 1 else "ham",
            "score": proba,
            "threshold": threshold,
            "explanation_type": "unavailable",
            "units": "probability",
            "explained_output": "P(smishing)",
            "base_value": None,
            "additivity_gap": None,
            "top_positive_features": [],
            "top_negative_features": [],
            "explanation_error": str(exc),
            "disclaimer": _DISCLAIMER,
        }

    row = _row_values(X_instance)

    # Rank features by absolute SHAP value
    contributions = sorted(
        zip(range(len(feature_columns)), feature_columns, shap_values),
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    def _make(ranked, direction: str) -> list[dict[str, Any]]:
        out = []
        for idx, (display_name, block, raw_name), value in ranked:
            # "increases risk" means a positive contribution to P(smishing).
            if direction == "increases_risk" and not value > 0:
                continue
            if direction == "decreases_risk" and not value < 0:
                continue
            out.append(
                {
                    "feature": display_name,
                    "shap_value": float(value),
                    "feature_value": row.get(idx, 0.0),
                    "direction": direction,
                    "description": (
                        f"TF-IDF weight of the {block} n-gram {raw_name!r} in this message"
                    ),
                }
            )
        return out[:_TOP_K]

    return {
        "model_name": model_name,
        "input_text": text[:200],
        "label": label,
        "label_name": "smishing" if label == 1 else "ham",
        "score": proba,
        "threshold": threshold,
        "explanation_type": explanation_type,
        "units": "probability",
        "explained_output": "P(smishing)",
        "base_value": base_value,
        "additivity_gap": gap,
        "top_positive_features": _make(contributions, "increases_risk"),
        "top_negative_features": _make(contributions, "decreases_risk"),
        "disclaimer": _DISCLAIMER,
    }


def _compute_shap_values(clf, X_instance, transform_fn, background_texts):
    """
    Explain ``clf.predict_proba`` - the model whose score is displayed.

    ``clf`` is the *fitted* classifier from the pipeline, calibration wrapper
    included. Unwrapping ``CalibratedClassifierCV`` (as the previous
    implementation did) would explain the uncalibrated inner ``LinearSVC``
    decision function, which is a different function from the probability the
    API returns.

    KernelExplainer is model-agnostic, so it works for any pipeline winner and
    returns values in probability units that sum to the displayed score.
    """
    if background_texts is None:
        background_texts = [_EMPTY_MESSAGE]
    bg_X = transform_fn(list(background_texts))

    class_index = positive_class_index(clf)

    with warnings.catch_warnings():
        # shap's internal lars_path/lstsq regressions emit ConvergenceWarning on
        # sparse TF-IDF matrices; they do not affect the returned values.
        warnings.simplefilter("ignore")
        # l1_reg=0 disables shap's Lasso-based attribution sparsification, which
        # is deprecated in shap 0.51 and can zero out genuine contributions.
        # SHAP's varying-feature optimisation already restricts the solve to the
        # ~hundreds of token columns that differ from the background.
        explainer = shap.KernelExplainer(clf.predict_proba, bg_X, l1_reg=0)
        raw = explainer.shap_values(X_instance, nsamples=_KERNEL_NSAMPLES, silent=True)

    shap_values = select_class_shap_values(raw, class_index)
    base_value = expected_value_for(explainer, class_index)
    model_output = float(np.asarray(clf.predict_proba(X_instance))[0][class_index])
    gap = additivity_gap(base_value, shap_values, model_output)

    return shap_values, "KernelExplainer", base_value, gap


_DISCLAIMER = (
    "This is a model-based risk assessment. SHAP values indicate feature influence "
    "on the model's prediction, not causal proof that a message is fraudulent. "
    "Verify important requests through official channels."
)
