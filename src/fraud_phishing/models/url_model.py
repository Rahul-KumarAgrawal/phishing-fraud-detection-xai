"""
url_model.py
============
URL phishing classification pipeline using extracted static features.

Compares:
  - Logistic Regression
  - Random Forest

Both operate on the 33-feature vector produced by
fraud_phishing.features.url_features.extract_feature_vector().

The fitted pipeline is saved to models/url_model.joblib together with its
metadata. Every loader verifies that metadata against the current extractor
(feature names, order, count, schema version and fingerprint) and refuses to
run an artifact it cannot check - see ``_assert_schema``.
"""

import hashlib
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

# The revised model: trained on the augmented pool (PHIUSIL + MURL) with a
# feature subset chosen by measured ablation (scripts/ablate_url_features.py).
# It is produced alongside the original model so the two can be compared
# side by side before anything is switched over.
URL_REVISED_MODEL_PATH = MODELS_DIR / "url_model_revised.joblib"
URL_REVISED_METADATA_PATH = MODELS_DIR / "url_model_revised_metadata.json"

# Identifier for the feature schema. Bump this whenever FEATURE_NAMES changes.
FEATURE_SCHEMA_VERSION = 2

# Metadata keys every URL model artifact must carry. A model whose metadata is
# missing one of these cannot be checked against the current extractor, so it is
# refused instead of being run blind.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "feature_names",
    "n_features",
    "feature_schema_version",
)

# A separate model trained on the domain-isolated pool. It exists so that
# "unseen domain" claims can be measured against a model that has genuinely
# never seen those registered domains (see reports/url_leakage_report.json).
URL_DOMAIN_MODEL_PATH = MODELS_DIR / "url_model_domain_holdout.joblib"
URL_DOMAIN_METADATA_PATH = MODELS_DIR / "url_model_domain_holdout_metadata.json"


def feature_schema_fingerprint(feature_names=None) -> str:
    """
    Stable short hash of the canonical URL feature schema.

    Written into every artifact's metadata by ``_stamp_schema`` on save and
    re-checked on load. Unlike ``feature_schema_version`` (bumped by hand), the
    fingerprint also catches a schema edit that was *not* version-bumped -
    including inserting a new feature in the middle, which shifts every later
    column while leaving the relative order of the old names intact.
    """
    names = FEATURE_NAMES if feature_names is None else tuple(feature_names)
    payload = f"v{FEATURE_SCHEMA_VERSION}|" + "|".join(names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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

def url_feature_matrix(urls: list[str], feature_names: list[str] | None = None) -> np.ndarray:
    """Alias kept for readability in the ablation/comparison scripts."""
    return urls_to_feature_matrix(urls, feature_names=feature_names)


def urls_to_feature_matrix(
    urls: list[str], feature_names: list[str] | None = None
) -> np.ndarray:
    """
    Convert a list of URL strings to a numeric feature matrix.

    Rows that fail feature extraction are replaced with zeros and logged. When
    ``feature_names`` is given, the matrix is restricted to those columns (in
    that order), which is how feature-subset models are trained and served.
    """
    rows = []
    for url in urls:
        try:
            rows.append(extract_feature_vector(url))
        except Exception as exc:
            print(f"  [WARN] Feature extraction failed for '{url[:80]}': {exc}")
            rows.append([0.0] * len(FEATURE_NAMES))
    X = np.array(rows, dtype=float)
    if feature_names is not None and list(feature_names) != list(FEATURE_NAMES):
        idx = feature_indices(feature_names)
        X = X[:, idx]
    return X


def feature_indices(feature_names: list[str]) -> list[int]:
    """Column indices of ``feature_names`` within the canonical FEATURE_NAMES order."""
    unknown = [n for n in feature_names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}")
    return [FEATURE_NAMES.index(n) for n in feature_names]


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
    URL_METADATA_PATH.write_text(
        json.dumps(_stamp_schema(metadata), indent=2), encoding="utf-8"
    )
    print(f"URL model saved : {URL_MODEL_PATH}")
    print(f"URL metadata    : {URL_METADATA_PATH}")


def _stamp_schema(metadata: dict) -> dict:
    """
    Record, in the metadata, the feature schema the artifact was trained on.

    Called by every ``save_*`` function so the recorded values cannot drift from
    the code that writes them. ``feature_names`` is preserved because
    feature-subset models legitimately use a subset; the rest is derived from it.
    """
    names = list(metadata.get("feature_names") or FEATURE_NAMES)
    metadata["feature_names"] = names
    metadata["n_features"] = len(names)
    metadata["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["feature_schema_fingerprint"] = feature_schema_fingerprint()
    return metadata


def _assert_schema(pipeline: Pipeline, metadata: dict, path: pathlib.Path) -> None:
    """
    Fail loudly if a saved model's metadata does not match the current extractor.

    Without this, a change to ``url_features.py`` silently shifts every column
    and the model keeps returning confident, wrong answers, while a missing
    metadata file silently drops the tuned threshold back to 0.5. Every check
    below is therefore fatal. Feature subsets are allowed (the revised model
    uses one) but must declare themselves with a ``feature_subset`` key and
    preserve the canonical relative order.
    """
    if not metadata:
        raise RuntimeError(
            f"Refusing to load {path.name}: its metadata is missing or empty. "
            "Expected the matching *_metadata.json beside the model. Retrain "
            "with: python scripts/train_url_model.py"
        )
    missing = [key for key in REQUIRED_METADATA_KEYS if not metadata.get(key)]
    if missing:
        raise RuntimeError(
            f"Refusing to load {path.name}: metadata is missing required key(s) "
            f"{missing}. Retrain with: python scripts/train_url_model.py"
        )

    names = list(metadata["feature_names"])
    if len(names) != len(set(names)):
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: metadata lists "
            "duplicate feature names. Retrain the model."
        )
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: unknown feature(s) "
            f"{unknown}. The extractor changed; retrain the model."
        )
    positions = [FEATURE_NAMES.index(n) for n in names]
    if positions != sorted(positions):
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: feature order differs "
            "from the canonical order in url_features.FEATURE_NAMES. Retrain the "
            "model."
        )

    try:
        declared_n = int(metadata["n_features"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Refusing to load {path.name}: metadata n_features is not an integer "
            f"({metadata['n_features']!r}). Retrain the model."
        ) from exc
    if declared_n != len(names):
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: metadata says "
            f"n_features={declared_n} but lists {len(names)} feature names. "
            "Retrain the model."
        )

    if tuple(names) != FEATURE_NAMES and not metadata.get("feature_subset"):
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: the feature list is not "
            "the current canonical schema, yet the metadata does not declare a "
            "feature_subset. The model was trained on a different feature schema; "
            "retrain it."
        )

    expected_width = getattr(pipeline, "n_features_in_", None)
    if expected_width is None:
        raise RuntimeError(
            f"Refusing to load {path.name}: the pipeline does not report "
            "n_features_in_, so its input width cannot be checked against its "
            "metadata. Re-export the model."
        )
    if int(expected_width) != len(names):
        raise RuntimeError(
            f"Feature-schema mismatch loading {path.name}: the pipeline takes "
            f"{expected_width} features but its metadata lists {len(names)}. "
            "Retrain the model with the current extractor."
        )

    version = metadata["feature_schema_version"]
    if version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Feature-schema version mismatch loading {path.name}: model says "
            f"{version}, code says {FEATURE_SCHEMA_VERSION}. Retrain the model."
        )

    # Optional for artifacts saved before P5; present in everything saved since.
    fingerprint = metadata.get("feature_schema_fingerprint")
    if fingerprint is not None and fingerprint != feature_schema_fingerprint():
        raise RuntimeError(
            f"Feature-schema fingerprint mismatch loading {path.name}: the model "
            f"was trained under {fingerprint!r} but the code now builds "
            f"{feature_schema_fingerprint()!r}. Retrain the model."
        )


def _load_artifact(
    model_path: pathlib.Path, metadata_path: pathlib.Path
) -> tuple[Pipeline, dict]:
    """
    Load a URL pipeline together with its metadata, verifying the schema first.

    The metadata file is required: without it the extracted feature order, the
    feature subset and the tuned threshold are all unknown, so the artifact is
    refused rather than run with 0.5 and the full schema.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run: python scripts/train_url_model.py"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Refusing to load {model_path.name}: its metadata file "
            f"{metadata_path.name} is missing, so the feature schema cannot be "
            "verified. Retrain with: python scripts/train_url_model.py"
        )
    pipeline = joblib.load(model_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Refusing to load {model_path.name}: {metadata_path.name} is not "
            f"valid JSON ({exc}). Retrain the model."
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Refusing to load {model_path.name}: {metadata_path.name} must "
            f"contain a JSON object, got {type(metadata).__name__}."
        )
    _assert_schema(pipeline, metadata, model_path)
    return pipeline, metadata


def load_model() -> tuple[Pipeline, dict]:
    return _load_artifact(URL_MODEL_PATH, URL_METADATA_PATH)


def save_revised_model(pipeline: Pipeline, metadata: dict) -> None:
    """Persist the feature-subset model trained on the augmented pool."""
    joblib.dump(pipeline, URL_REVISED_MODEL_PATH)
    URL_REVISED_METADATA_PATH.write_text(
        json.dumps(_stamp_schema(metadata), indent=2), encoding="utf-8"
    )
    print(f"Revised model    : {URL_REVISED_MODEL_PATH}")
    print(f"Revised metadata : {URL_REVISED_METADATA_PATH}")


def load_revised_model() -> tuple[Pipeline, dict]:
    return _load_artifact(URL_REVISED_MODEL_PATH, URL_REVISED_METADATA_PATH)


def save_domain_holdout_model(pipeline: Pipeline, metadata: dict) -> None:
    """Persist the model trained on the domain-isolated pool (evaluation only)."""
    joblib.dump(pipeline, URL_DOMAIN_MODEL_PATH)
    URL_DOMAIN_METADATA_PATH.write_text(
        json.dumps(_stamp_schema(metadata), indent=2), encoding="utf-8"
    )
    print(f"Domain-holdout model    : {URL_DOMAIN_MODEL_PATH}")
    print(f"Domain-holdout metadata : {URL_DOMAIN_METADATA_PATH}")


def load_domain_holdout_model() -> tuple[Pipeline, dict]:
    """
    Load the model trained on the domain-isolated pool.

    This model, not the production model, is the one that may be evaluated on
    ``urls_domain_test.csv`` / ``urls_template_test.csv``: the production model
    is trained on a random split that shares most of those test URLs (see
    reports/url_leakage_report.json).
    """
    return _load_artifact(URL_DOMAIN_MODEL_PATH, URL_DOMAIN_METADATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_url(
    url: str,
    pipeline: Pipeline,
    threshold: float = 0.5,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run inference on a single URL string.

    ``feature_names`` must match the list the pipeline was trained on. It
    defaults to the full canonical schema.

    Returns dict with:
      label, label_name, score, threshold, features (dict of extracted values)
    """
    if not url or not url.strip():
        raise ValueError("URL must not be empty.")

    feat_dict = extract_features(url)
    names = list(FEATURE_NAMES) if feature_names is None else list(feature_names)
    X_mat = np.array([[feat_dict[n] for n in names]], dtype=float)
    proba = pipeline.predict_proba(X_mat)[0][1]
    label = int(proba >= threshold)

    return {
        "label": label,
        "label_name": "phishing" if label == 1 else "legitimate",
        "score": float(proba),
        "threshold": threshold,
        "features": feat_dict,
    }
