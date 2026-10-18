"""
url_model.py
============
URL phishing classification pipeline using extracted static features.

Compares:
  - Logistic Regression
  - Random Forest

Both operate on the 30-feature vector produced by
fraud_phishing.features.url_features.extract_feature_vector().

The fitted pipeline is saved to models/url_model.joblib.
"""

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_phishing.features.url_features import (
    FEATURE_NAMES,
    extract_feature_vector,
    extract_features,
)

MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

URL_MODEL_PATH = MODELS_DIR / "url_model.joblib"
URL_METADATA_PATH = MODELS_DIR / "url_model_metadata.json"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            # RF does not require scaling but we include it for API consistency
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


CANDIDATE_PIPELINES: dict[str, Any] = {
    "LogisticRegression": build_lr_pipeline,
    "RandomForest": build_rf_pipeline,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def urls_to_feature_matrix(urls: list[str]) -> np.ndarray:
    """
    Convert a list of URL strings to a numeric feature matrix.
    Rows that fail feature extraction are replaced with zeros and logged.
    """
    rows = []
    for url in urls:
        try:
            rows.append(extract_feature_vector(url))
        except Exception as exc:
            print(f"  [WARN] Feature extraction failed for '{url[:80]}': {exc}")
            rows.append([0.0] * len(FEATURE_NAMES))
    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UrlEvalResult:
    model_name: str
    f1: float
    roc_auc: float
    report: str
    confusion: list[list[int]]
    threshold: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate_pipeline(
    pipeline: Pipeline,
    X_mat: np.ndarray,
    y: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> UrlEvalResult:
    proba = pipeline.predict_proba(X_mat)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y, y_pred, target_names=["legitimate", "phishing"])
    cm = confusion_matrix(y, y_pred).tolist()
    f1 = f1_score(y, y_pred)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")

    return UrlEvalResult(
        model_name=model_name,
        f1=f1,
        roc_auc=auc,
        report=report,
        confusion=cm,
        threshold=threshold,
    )


def select_threshold(pipeline: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> float:
    proba = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    print(f"  Selected threshold: {best_threshold:.4f} (val F1={f1_scores[best_idx]:.4f})")
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, metadata: dict) -> None:
    joblib.dump(pipeline, URL_MODEL_PATH)
    URL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"URL model saved : {URL_MODEL_PATH}")
    print(f"URL metadata    : {URL_METADATA_PATH}")


def load_model() -> tuple[Pipeline, dict]:
    if not URL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"URL model not found at {URL_MODEL_PATH}. "
            "Run: python scripts/train_url_model.py"
        )
    pipeline = joblib.load(URL_MODEL_PATH)
    metadata: dict = {}
    if URL_METADATA_PATH.exists():
        metadata = json.loads(URL_METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_url(
    url: str,
    pipeline: Pipeline,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run inference on a single URL string.

    Returns dict with:
      label, label_name, score, threshold, features (dict of extracted values)
    """
    if not url or not url.strip():
        raise ValueError("URL must not be empty.")

    feat_dict = extract_features(url)
    X_mat = np.array([[feat_dict[name] for name in FEATURE_NAMES]], dtype=float)
    proba = pipeline.predict_proba(X_mat)[0][1]
    label = int(proba >= threshold)

    return {
        "label": label,
        "label_name": "phishing" if label == 1 else "legitimate",
        "score": float(proba),
        "threshold": threshold,
        "features": feat_dict,
    }
