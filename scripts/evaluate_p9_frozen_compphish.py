"""Perform the one fixed P9 evaluation of the P7 Random Forest candidate.

No fitting, threshold selection, model comparison, URL fetch, or frozen-input
mutation is allowed in this evaluation path.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score, roc_auc_score)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.features import url_features  # noqa: E402
from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: E402
from fraud_phishing.models.url_model import _load_artifact, urls_to_feature_matrix  # noqa: E402

FROZEN = ROOT / "data" / "processed" / "compphish_external_frozen.csv"
MANIFEST_PATH = ROOT / "reports" / "p6" / "compphish_external_frozen_manifest.json"
P7_REPORT = ROOT / "reports" / "p7" / "p7_internal_validation.json"
MODEL = ROOT / "models" / "p7_url_randomforest.joblib"
METADATA = ROOT / "models" / "p7_url_randomforest_metadata.json"
OUT = ROOT / "reports" / "p9"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Calculate P9 metrics for class 1 = phishing at a fixed threshold."""
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate_legitimate": float(fp / max(tn + fp, 1)),
        "false_negative_rate_phishing": float(fn / max(fn + tp, 1)),
        "predicted_class_counts": {"legitimate": int((predicted == 0).sum()), "phishing": int((predicted == 1).sum())},
        "actual_class_counts": {"legitimate": int((y == 0).sum()), "phishing": int((y == 1).sum())},
        "by_actual_class": {
            "legitimate": {"rows": int((y == 0).sum()), "predicted_legitimate": int(((y == 0) & (predicted == 0)).sum()), "predicted_phishing": int(((y == 0) & (predicted == 1)).sum())},
            "phishing": {"rows": int((y == 1).sum()), "predicted_legitimate": int(((y == 1) & (predicted == 0)).sum()), "predicted_phishing": int(((y == 1) & (predicted == 1)).sum())},
        },
        "probability_distribution": {"min": float(scores.min()), "p01": float(np.quantile(scores, .01)), "p05": float(np.quantile(scores, .05)), "p25": float(np.quantile(scores, .25)), "median": float(np.median(scores)), "p75": float(np.quantile(scores, .75)), "p95": float(np.quantile(scores, .95)), "p99": float(np.quantile(scores, .99)), "max": float(scores.max()), "mean": float(scores.mean()), "std": float(scores.std()), "by_actual_class": {str(label): {"rows": int((y == label).sum()), "mean": float(scores[y == label].mean()), "median": float(np.median(scores[y == label])), "p05": float(np.quantile(scores[y == label], .05)), "p95": float(np.quantile(scores[y == label], .95))} for label in (0, 1)}},
    }


def verify_inputs() -> tuple[pd.DataFrame, dict, object, dict, dict]:
    """Verify every immutable evaluation input before constructing features."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Frozen manifest is not complete")
    if sha256(FROZEN) != manifest["output"]["sha256"]:
        raise RuntimeError("Frozen CSV SHA-256 differs from the manifest; refusing inference")
    expected = "56f6e8545a7df5f675b740d0858ee0fb4b2fa920a085d4aceae2838bfbc0e377"
    if manifest["output"]["sha256"] != expected:
        raise RuntimeError("Frozen manifest does not contain the approved P9 CSV hash")
    if manifest.get("post_filter_overlap") != {"exact_url": 0, "normalized_url": 0, "hostname": 0, "registered_domain": 0, "template": 0}:
        raise RuntimeError("Frozen manifest does not prove zero development overlap")
    frame = pd.read_csv(FROZEN)
    # The freeze retains CompPhish's source serial number for provenance, plus
    # the raw URL and binary label used by the evaluator.
    if list(frame.columns) != ["serial_number", "url", "label"] or frame.url.isna().any() or not set(frame.label.unique()).issubset({0, 1}):
        raise RuntimeError("Frozen CSV schema or labels are invalid")
    retained = manifest["counts"]["retained_class_counts"]
    actual = {str(key): int(value) for key, value in frame.label.value_counts().sort_index().items()}
    if len(frame) != manifest["counts"]["retained_rows"] or actual != retained:
        raise RuntimeError("Frozen CSV counts differ from manifest")
    model, metadata = _load_artifact(MODEL, METADATA)
    p7 = json.loads(P7_REPORT.read_text(encoding="utf-8"))["candidates"]["RandomForest"]
    if sha256(MODEL) != p7["artifact_sha256"] or sha256(METADATA) != p7["metadata_sha256"]:
        raise RuntimeError("P7 candidate artifact or metadata hash differs from the recorded P7 report")
    if metadata.get("label_map") != {"0": "legitimate", "1": "phishing"}:
        raise RuntimeError("P7 candidate does not use the required label mapping")
    if list(metadata.get("feature_names", [])) != list(FEATURE_NAMES) or int(metadata.get("n_features", -1)) != len(FEATURE_NAMES):
        raise RuntimeError("P7 candidate schema differs from the canonical extractor")
    if list(model.classes_) != [0, 1]:
        raise RuntimeError("P7 candidate class order is not [0, 1]")
    return frame, manifest, model, metadata, p7


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Pin public-suffix parsing to the same offline cache approach used in P7.
    url_features.tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=str(ROOT / "reports" / "p6" / ".tldextract"))
    before_hash = sha256(FROZEN)
    frame, manifest, model, metadata, p7 = verify_inputs()
    X = urls_to_feature_matrix(frame.url.astype(str).tolist())
    scores = model.predict_proba(X)[:, 1]
    result = compute_metrics(frame.label.to_numpy(dtype=int), scores, float(metadata["threshold"]))
    after_hash = sha256(FROZEN)
    if before_hash != after_hash:
        raise RuntimeError("Frozen CSV changed during evaluation; refusing to save a result")
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation": "single fixed P9 evaluation of P7 RandomForest only", "input_verification": {"frozen_csv": str(FROZEN.relative_to(ROOT)), "frozen_sha256_before": before_hash, "frozen_sha256_after": after_hash, "manifest": str(MANIFEST_PATH.relative_to(ROOT)), "manifest_source": manifest["source"], "manifest_label_convention": "0=legitimate, 1=phishing", "manifest_retained_counts": manifest["counts"]["retained_class_counts"], "manifest_post_filter_overlap": manifest["post_filter_overlap"], "candidate": str(MODEL.relative_to(ROOT)), "candidate_sha256": sha256(MODEL), "candidate_metadata_sha256": sha256(METADATA), "feature_schema_fingerprint": metadata["feature_schema_fingerprint"], "feature_count": metadata["n_features"], "class_order": [int(x) for x in model.classes_]}, "operating_point": {"threshold": float(metadata["threshold"]), "threshold_source": metadata["threshold_selection"], "no_retraining_refitting_recalibration_or_tuning": True}, "metrics": result, "external_validity": {"exclusions": "Rows overlapping the P6 development boundary at exact URL, conservative normalized URL, hostname, registered domain, or normalized template were excluded; each post-filter overlap count is zero.", "class_balance_change": {"pre_filter_unique": manifest["counts"]["original_unique_class_counts"], "retained": manifest["counts"]["retained_class_counts"]}, "prior_exposure": manifest["exposure_before_freeze"], "limitations": ["CompPhish structural distributions were examined in P6.1 before the candidate was evaluated, so this is not fully blind external validation.", "Strict overlap exclusions changed both the class balance and the retained URL population; results apply to this filtered subset, not all CompPhish or real-world traffic.", "CompPhish provenance and collection choices can impose source-selection bias; this one dataset is not perfectly representative of real-world traffic.", "The frozen result must not be used for further iterative candidate selection or tuning."]}}
    (OUT / "p9_frozen_compphish_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": result, "report": str((OUT / "p9_frozen_compphish_metrics.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
