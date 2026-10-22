"""Train P7 candidate-only URL baselines on the prepared development split."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.features import url_features  # noqa: E402
from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: E402
from fraud_phishing.models.url_model import CANDIDATE_PIPELINES, _load_artifact, _stamp_schema, select_threshold, urls_to_feature_matrix  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "p7"
MODELS = ROOT / "models"
SEED = 42
MAX_FIT_ROWS = 150_000


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def fit_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_FIT_ROWS: return frame
    rng = np.random.default_rng(SEED)
    pieces = []
    for label, part in frame.groupby("label", sort=True):
        n = round(MAX_FIT_ROWS * len(part) / len(frame))
        pieces.append(part.iloc[np.sort(rng.choice(len(part), n, replace=False))])
    return pd.concat(pieces).sample(frac=1, random_state=SEED).reset_index(drop=True)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "false_positives": int(fp), "false_negatives": int(fn), "false_positive_rate_legitimate": float(fp / max(tn + fp, 1)), "false_negative_rate_phishing": float(fn / max(fn + tp, 1)), "validation_rows": int(len(y)), "validation_legitimate": int((y == 0).sum()), "validation_phishing": int((y == 1).sum())}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Avoid suffix-list refreshes; this changes no feature definition.
    url_features.tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=str(ROOT / "reports" / "p6" / ".tldextract"))
    train = pd.read_csv(DATA / "p7_url_train.csv")
    val = pd.read_csv(DATA / "p7_url_val.csv")
    fit = fit_sample(train)
    X_fit = urls_to_feature_matrix(fit.url.astype(str).tolist())
    X_val = urls_to_feature_matrix(val.url.astype(str).tolist())
    y_fit, y_val = fit.label.to_numpy(dtype=int), val.label.to_numpy(dtype=int)
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "P7 internal validation only; candidates are not served and frozen CompPhish was not read.", "schema": {"feature_names": list(FEATURE_NAMES), "n_features": len(FEATURE_NAMES)}, "fit_sample": {"rows": int(len(fit)), "class_counts": {str(k): int(v) for k,v in fit.label.value_counts().sort_index().items()}, "selection": "deterministic stratified subsample of leakage-isolated training partition, seed 42"}, "validation": {"rows": int(len(val)), "class_counts": {str(k): int(v) for k,v in val.label.value_counts().sort_index().items()}}, "candidates": {}}
    for name, factory in CANDIDATE_PIPELINES.items():
        pipe = factory(); pipe.fit(X_fit, y_fit)
        if list(pipe.classes_) != [0, 1]: raise RuntimeError(f"{name} class order is {pipe.classes_}")
        threshold = float(select_threshold(pipe, X_val, y_val))
        score = pipe.predict_proba(X_val)[:, 1]
        result = metrics(y_val, score, threshold)
        path = MODELS / f"p7_url_{name.lower()}.joblib"
        meta_path = MODELS / f"p7_url_{name.lower()}_metadata.json"
        metadata = _stamp_schema({"model_name": name, "model_variant": "p7_candidate", "threshold": threshold, "label_map": {"0": "legitimate", "1": "phishing"}, "feature_names": list(FEATURE_NAMES), "random_seed": SEED, "purpose": "P7 candidate only; not served or promoted", "training_data": "p7_url_train.csv deterministic fit subsample; PHIUSIL + conflict-quarantined MURL", "validation_data": "p7_url_val.csv connected domain/template held-out partition", "threshold_selection": "max validation F1", "validation_metrics": result, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        joblib.dump(pipe, path); meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _load_artifact(path, meta_path)
        report["candidates"][name] = {"artifact": str(path.relative_to(ROOT)), "artifact_sha256": sha256(path), "metadata": str(meta_path.relative_to(ROOT)), "metadata_sha256": sha256(meta_path), "threshold": threshold, **result}
    (REPORTS / "p7_internal_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
