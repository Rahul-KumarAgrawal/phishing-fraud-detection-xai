"""
shap_utils.py
=============
Shared helpers for keeping SHAP explanations aligned with the prediction the
API and UI actually display.

Two invariants are enforced here rather than assumed:

1. **Class alignment** - contributions explain the *positive* class
   (``1`` = phishing / smishing), which is the class whose probability the API
   returns. The positive-class column is resolved from the fitted estimator's
   ``classes_`` attribute instead of being hard-coded to index 1, so a model
   whose classes are stored in a different order cannot silently produce
   explanations for the wrong class.

2. **Additivity** - ``base_value + sum(shap_values)`` equals the explained model
   output for that input. :func:`additivity_gap` exposes the residual so callers
   (and the regression tests) can verify the invariant instead of trusting it.
"""

from __future__ import annotations

import numpy as np

# The project's positive class: 0 = legitimate / ham, 1 = phishing / smishing.
POSITIVE_CLASS = 1


def positive_class_index(clf) -> int:
    """
    Column index of the positive class in a fitted classifier's output.

    Falls back to 1 (the historical assumption) when ``classes_`` is absent,
    and to the last column when the positive class is not present at all.
    """
    classes = getattr(clf, "classes_", None)
    if classes is None:
        return 1

    classes = np.asarray(classes)
    matches = np.flatnonzero(classes == POSITIVE_CLASS)
    if matches.size == 0:
        return len(classes) - 1
    return int(matches[0])


def expected_value_for(explainer, class_index: int) -> float:
    """
    Scalar ``base_value`` for one class.

    Explainer APIs differ: TreeExplainer returns one expected value per class,
    LinearExplainer returns a scalar for binary tasks, and KernelExplainer
    returns one entry per output.
    """
    expected = np.asarray(getattr(explainer, "expected_value", 0.0), dtype=float).ravel()
    if expected.size == 0:
        return 0.0
    if expected.size == 1:
        return float(expected[0])
    if 0 <= class_index < expected.size:
        return float(expected[class_index])
    return float(expected[-1])


def _first_instance_row(values) -> np.ndarray:
    """Reduce a per-class SHAP array to a single 1-D vector for one instance."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    return np.asarray(arr[0], dtype=float).reshape(-1)


def select_class_shap_values(shap_values, class_index: int) -> np.ndarray:
    """
    Extract the SHAP values for one instance and one class.

    Handles every shape the installed SHAP versions may return for a binary
    classifier:

    * ``list`` of per-class arrays (``shap_values[class]``)
    * 3-D array ``(n_samples, n_features, n_classes)``
    * 2-D array ``(n_samples, n_features)`` - already the positive-class output
      of a binary linear model
    * 1-D array ``(n_features,)``
    """
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("Empty SHAP value list")
        index = class_index if 0 <= class_index < len(shap_values) else len(shap_values) - 1
        return _first_instance_row(shap_values[index])

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        if arr.shape[2] > 1 and 0 <= class_index < arr.shape[2]:
            return np.asarray(arr[0, :, class_index], dtype=float).reshape(-1)
        return np.asarray(arr[0, :, 0], dtype=float).reshape(-1)
    if arr.ndim == 2:
        return np.asarray(arr[0], dtype=float).reshape(-1)
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def additivity_gap(base_value: float, shap_values: np.ndarray, model_output: float) -> float:
    """
    Residual ``|base + sum(shap) - model_output|``.

    ~0 for an exact explainer on the function it was fitted to. A large value
    means the explanation does not describe the displayed score, which is the
    defect this module exists to prevent.
    """
    return float(abs(float(base_value) + float(np.sum(shap_values)) - float(model_output)))
