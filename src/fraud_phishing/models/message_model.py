"""
message_model.py
================
Message classification pipeline using TF-IDF + scikit-learn classifiers.

Compares:
  - Logistic Regression
  - Linear SVM (LinearSVC)
  - Multinomial Naive Bayes

Each model is wrapped in a scikit-learn Pipeline with:
  FeatureUnion([
      word-level TF-IDF (1-3 grams),
      character-level TF-IDF (2-5 grams),
  ])

The fitted pipeline is saved with joblib to models/message_model.joblib together
with its metadata. ``load_model`` refuses an artifact whose metadata is missing
or whose feature specification no longer matches ``FEATURE_SPEC``, so a stale
file cannot silently run with the default 0.5 threshold.
"""

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── Model artefact paths ───────────────────────────────────────────────────
MESSAGE_MODEL_PATH = MODELS_DIR / "message_model.joblib"
MESSAGE_METADATA_PATH = MODELS_DIR / "message_model_metadata.json"

# Canonical description of the TF-IDF feature space. Single source of truth:
# written into the artifact metadata by scripts/train_message_model.py and
# re-checked when the artifact is loaded. Change it only together with
# ``_make_tfidf_union`` and retrain.
FEATURE_SPEC = "TF-IDF word(1-3gram) + char_wb(2-5gram)"

# Metadata keys the runtime depends on. Missing any of them means the artifact
# cannot be interpreted safely: without ``threshold`` the decision silently
# falls back to 0.5 and borderline messages are mislabelled.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "model_name",
    "threshold",
    "label_map",
    "dataset_provenance",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_tfidf_union(
    word_max_features: int = 50_000,
    char_max_features: int = 30_000,
) -> FeatureUnion:
    """Combined word + character n-gram TF-IDF feature union."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=word_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        decode_error="replace",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=char_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        decode_error="replace",
    )
    return FeatureUnion(
        [("word", word_tfidf), ("char", char_tfidf)],
        n_jobs=1,
    )


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", _make_tfidf_union()),
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


def build_svm_pipeline() -> Pipeline:
    """LinearSVC wrapped in CalibratedClassifierCV for probability support."""
    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    calibrated_svm = CalibratedClassifierCV(svm, cv=3, method="sigmoid")
    return Pipeline(
        [
            ("features", _make_tfidf_union()),
            ("clf", calibrated_svm),
        ]
    )


def build_nb_pipeline() -> Pipeline:
    """MultinomialNB – requires non-negative features (TF-IDF ≥ 0 ✓)."""
    return Pipeline(
        [
            ("features", _make_tfidf_union()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )


CANDIDATE_PIPELINES: dict[str, Pipeline] = {
    "LogisticRegression": build_lr_pipeline,
    "LinearSVC": build_svm_pipeline,
    "MultinomialNB": build_nb_pipeline,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name: str
    f1: float
    roc_auc: float
    report: str
    confusion: list[list[int]]
    threshold: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> EvalResult:
    """Evaluate a fitted pipeline on provided X/y."""
    proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y, y_pred, target_names=["ham", "smishing"])
    cm = confusion_matrix(y, y_pred).tolist()
    f1 = f1_score(y, y_pred)

    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")

    return EvalResult(
        model_name=model_name,
        f1=f1,
        roc_auc=auc,
        report=report,
        confusion=cm,
        threshold=threshold,
        extra={"proba_sample": proba[:5].tolist()},
    )


def select_threshold(pipeline: Pipeline, X_val: pd.Series, y_val: pd.Series) -> float:
    """
    Select the classification threshold that maximises F1 on the validation set.
    Only validation data is used – no test leakage.
    """
    proba = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last element is threshold-free
    best_threshold = float(thresholds[best_idx])
    print(f"  Selected threshold: {best_threshold:.4f} (val F1={f1_scores[best_idx]:.4f})")
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, metadata: dict) -> None:
    metadata.setdefault("feature_type", FEATURE_SPEC)
    joblib.dump(pipeline, MESSAGE_MODEL_PATH)
    MESSAGE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model saved: {MESSAGE_MODEL_PATH}")
    print(f"Metadata  : {MESSAGE_METADATA_PATH}")


def _assert_artifact(pipeline: Pipeline, metadata: dict, path: pathlib.Path) -> None:
    """
    Fail loudly if a message artifact is missing metadata or disagrees with code.

    The TF-IDF vocabularies travel inside the pipeline, so a stale artifact
    cannot silently shift columns the way a URL model can. What it *can* do is
    lose its tuned threshold (the loader would fall back to 0.5) or be trained
    under a different feature specification, so both are checked here.
    """
    if not metadata:
        raise RuntimeError(
            f"Refusing to load {path.name}: its metadata is missing or empty. "
            f"Expected {MESSAGE_METADATA_PATH.name} beside the model. Retrain "
            "with: python scripts/train_message_model.py"
        )
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise RuntimeError(
            f"Refusing to load {path.name}: metadata is missing required key(s) "
            f"{missing}. Retrain with: python scripts/train_message_model.py"
        )

    threshold = metadata["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise RuntimeError(
            f"Refusing to load {path.name}: metadata threshold {threshold!r} is not "
            "a number. Retrain with: python scripts/train_message_model.py"
        )
    if not 0.0 < float(threshold) < 1.0:
        raise RuntimeError(
            f"Refusing to load {path.name}: metadata threshold {threshold!r} is not "
            "a probability in (0, 1). Retrain with: python "
            "scripts/train_message_model.py"
        )

    spec = metadata.get("feature_type")
    if spec != FEATURE_SPEC:
        raise RuntimeError(
            f"Feature-spec mismatch loading {path.name}: the artifact was trained "
            f"with {spec!r} but the code builds {FEATURE_SPEC!r}. Retrain with: "
            "python scripts/train_message_model.py"
        )

    steps = getattr(pipeline, "named_steps", {})
    for step in ("features", "clf"):
        if step not in steps:
            raise RuntimeError(
                f"Refusing to load {path.name}: the pipeline has no {step!r} step."
            )
    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError(
            f"Refusing to load {path.name}: the pipeline has no predict_proba, so "
            "no calibrated probability can be produced."
        )


def load_model() -> tuple[Pipeline, dict]:
    """Load the saved message pipeline and its metadata."""
    if not MESSAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Message model not found at {MESSAGE_MODEL_PATH}. "
            "Run: python scripts/train_message_model.py"
        )
    if not MESSAGE_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Refusing to load {MESSAGE_MODEL_PATH.name}: its metadata file "
            f"{MESSAGE_METADATA_PATH.name} is missing, so the tuned threshold and "
            "feature spec are unknown. Retrain with: python "
            "scripts/train_message_model.py"
        )
    pipeline = joblib.load(MESSAGE_MODEL_PATH)
    try:
        metadata = json.loads(MESSAGE_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Refusing to load {MESSAGE_MODEL_PATH.name}: "
            f"{MESSAGE_METADATA_PATH.name} is not valid JSON ({exc}). Retrain the "
            "model."
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Refusing to load {MESSAGE_MODEL_PATH.name}: "
            f"{MESSAGE_METADATA_PATH.name} must contain a JSON object, got "
            f"{type(metadata).__name__}."
        )
    _assert_artifact(pipeline, metadata, MESSAGE_MODEL_PATH)
    return pipeline, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_message(
    text: str,
    pipeline: Pipeline,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run inference on a single raw text string.

    Parameters
    ----------
    text : str
        Raw message text.
    pipeline : Pipeline
        Fitted scikit-learn pipeline (must have predict_proba).
    threshold : float
        Decision threshold (default 0.5, use metadata threshold in production).

    Returns
    -------
    dict with keys: label (int), label_name (str), score (float)
    """
    if not text or not text.strip():
        raise ValueError("Message text must not be empty.")

    proba = pipeline.predict_proba([text])[0][1]
    label = int(proba >= threshold)
    return {
        "label": label,
        "label_name": "smishing" if label == 1 else "ham",
        "score": float(proba),
        "threshold": threshold,
    }
