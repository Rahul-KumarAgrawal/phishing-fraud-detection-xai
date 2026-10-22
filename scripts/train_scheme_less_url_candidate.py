"""Train a non-served candidate that distinguishes an absent scheme from HTTP.

This experiment appends ``has_explicit_scheme`` after the existing canonical
feature subset. It never alters the served artifact, its metadata, threshold,
or the rollback pair. The train/validation/test partitions are the already
frozen showcase development splits.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_phishing.features.url_features import FEATURE_NAMES, _URL_SCHEME_RE  # noqa: E402
from fraud_phishing.models.url_model import CANDIDATE_PIPELINES, _stamp_schema, select_threshold, urls_to_feature_matrix  # noqa: E402
from train_showcase_url_candidates import metrics  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "showcase" / "scheme_less_candidate_results.json"
FEATURE_NAMES_CANDIDATE = [
    name for name in FEATURE_NAMES
    if name not in {"brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain"}
] + ["has_explicit_scheme"]
FIT_ROWS = 12_000
EVALUATION_ROWS = 4_000


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False) as handle:
        temp = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2)
    try:
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def candidate_matrix(urls: list[str]) -> np.ndarray:
    full = urls_to_feature_matrix(urls)
    base = full[:, [FEATURE_NAMES.index(name) for name in FEATURE_NAMES_CANDIDATE[:-1]]]
    explicit = np.array([int(bool(_URL_SCHEME_RE.match(url.strip()))) for url in urls], dtype=float).reshape(-1, 1)
    return np.hstack((base, explicit))


def deterministic_stratified_sample(frame: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """Bound this diagnostic candidate without changing its grouped source split."""
    if len(frame) <= total_rows:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(42)
    parts = []
    for label in (0, 1):
        group = frame[frame.label == label].sort_values("url", kind="stable")
        take = min(len(group), total_rows // 2)
        selected = np.sort(rng.choice(len(group), take, replace=False))
        parts.append(group.iloc[selected])
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def probe(model, threshold: float, url: str) -> dict:
    X = candidate_matrix([url])
    score = float(model.predict_proba(X)[0][1])
    return {"url": url, "score": score, "label": int(score >= threshold), "has_explicit_scheme": int(X[0, -1])}


def main() -> None:
    train = pd.read_csv(DATA / "showcase_url_train.csv")
    validation = pd.read_csv(DATA / "showcase_url_validation.csv")
    test = pd.read_csv(DATA / "showcase_url_test.csv")
    fit = deterministic_stratified_sample(train, FIT_ROWS)
    validation = deterministic_stratified_sample(validation, EVALUATION_ROWS)
    test = deterministic_stratified_sample(test, EVALUATION_ROWS)
    print("Extracting static features for the isolated scheme-handling candidate...", flush=True)
    X_fit = candidate_matrix(fit.url.astype(str).tolist())
    X_validation = candidate_matrix(validation.url.astype(str).tolist())
    X_test = candidate_matrix(test.url.astype(str).tolist())
    y_fit = fit.label.to_numpy(dtype=int)
    y_validation = validation.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)

    print("Fitting candidate...", flush=True)
    model = CANDIDATE_PIPELINES["RandomForest"]().fit(X_fit, y_fit)
    threshold = float(select_threshold(model, X_validation, y_validation))
    validation_scores = model.predict_proba(X_validation)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    artifact = MODELS / "scheme_less_randomforest_candidate.joblib"
    metadata_path = MODELS / "scheme_less_randomforest_candidate_metadata.json"
    joblib.dump(model, artifact)
    metadata = _stamp_schema({
        "model_name": "RandomForest",
        "model_variant": "scheme_less_candidate_not_served",
        "feature_configuration": "p8_without_brand_plus_has_explicit_scheme",
        "feature_names": FEATURE_NAMES_CANDIDATE,
        "threshold": threshold,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "purpose": "Candidate only; distinguishes absent scheme from explicit HTTP without changing the served model",
        "training_data": "showcase_url_train.csv deterministic source-aware fit sample",
        "threshold_selection": "maximum F1 on grouped validation partition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(metadata, metadata_path)
    report = {
        "status": "complete",
        "served_model_changed": False,
        "canonical_feature_order_preserved_as_prefix": list(FEATURE_NAMES_CANDIDATE[:-1]) == [name for name in FEATURE_NAMES if name not in {"brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain"}],
        "new_candidate_feature": "has_explicit_scheme",
        "sampling": {"fit_rows": len(fit), "validation_rows": len(validation), "test_rows": len(test), "rule": "deterministic balanced subsamples within the existing grouped partitions"},
        "feature_names": FEATURE_NAMES_CANDIDATE,
        "threshold": threshold,
        "validation": metrics(y_validation, validation_scores, threshold),
        "test": metrics(y_test, test_scores, threshold),
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_sha256": sha256(artifact),
        "metadata": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha256(metadata_path),
        "probes": [probe(model, threshold, url) for url in ["www.google.com", "google.com", "https://www.google.com/", "http://www.google.com/", "https://www.wikipedia.org/", "http://paypal-secure-login.example/verify?urgent=true"]],
        "limitations": "The candidate uses only existing development data and is not an external evaluation. It is retained separately and is not integrated automatically.",
    }
    atomic_json(report, REPORT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
