"""Run P8 development-only URL feature ablations and diagnostics.

This script intentionally reads only the P7 train/validation partitions.  In
particular, it never opens the frozen CompPhish CSV or its manifest.  It uses
static parsing only; URLs are never fetched, resolved, or visited.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import urllib.parse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_compphish_overlap import normalized_url  # noqa: E402
from audit_url_datasets import _registered_domain  # noqa: E402
from fraud_phishing.data.prepare_urls import url_template  # noqa: E402
from fraud_phishing.features import url_features  # noqa: E402
from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: E402
from fraud_phishing.models.url_model import (  # noqa: E402
    CANDIDATE_PIPELINES, _load_artifact, _stamp_schema, feature_indices,
    select_threshold, urls_to_feature_matrix,
)
from train_p7_url_candidates import fit_sample  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "p8"
MODELS = ROOT / "models"
SEED = 42
CV_ROWS = 60_000
CV_FOLDS = 3

# Each ablation removes one interpretable group while preserving the canonical
# order of all remaining columns.  No new feature is introduced in P8.
GROUPS: dict[str, tuple[str, ...]] = {
    "all_features": (),
    "without_transport": ("is_https",),
    "without_path_query": ("path_length", "slash_count", "path_depth", "query_length", "question_count", "equals_count", "ampersand_count", "query_param_count", "has_query"),
    "without_length_character": ("url_length", "hostname_length", "registered_domain_length", "dot_count", "hyphen_count", "slash_count", "underscore_count", "at_count", "question_count", "equals_count", "ampersand_count", "percent_count", "digit_count", "digit_ratio", "special_char_ratio", "tilde_count"),
    "without_hostname_subdomain": ("hostname_length", "registered_domain_length", "subdomain_count", "subdomain_length", "tld_length"),
    "without_suspicious_tokens": ("suspicious_token_count",),
    "without_brand": ("brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain"),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_bundle(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positives": int(fp), "false_negatives": int(fn),
        "false_positive_rate_legitimate": float(fp / max(tn + fp, 1)),
        "false_negative_rate_phishing": float(fn / max(fn + tp, 1)),
        "rows": int(len(y)), "legitimate": int((y == 0).sum()), "phishing": int((y == 1).sum()),
    }


def subset_names(name: str) -> list[str]:
    drop = set(GROUPS[name])
    return [feature for feature in FEATURE_NAMES if feature not in drop]


def deterministic_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pieces = []
    for label, part in frame.groupby("label", sort=True):
        size = round(max_rows * len(part) / len(frame))
        pieces.append(part.iloc[np.sort(rng.choice(len(part), size, replace=False))])
    return pd.concat(pieces).sample(frac=1, random_state=SEED).reset_index(drop=True)


def component_groups(frame: pd.DataFrame) -> np.ndarray:
    """Connected components induced by registered domain or URL template."""
    parent = list(range(len(frame)))
    size = [1] * len(frame)
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            if size[a] < size[b]: a, b = b, a
            parent[b] = a; size[a] += size[b]
    seen: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(frame.url.astype(str)):
        for key in (("domain", _registered_domain(raw)), ("template", url_template(raw))):
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    return np.asarray([find(i) for i in range(len(frame))])


def structural_frame(frame: pd.DataFrame, X: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Static subgroup fields for error analysis; no URL destination access."""
    out = frame[["url", "label", "source"]].copy().reset_index(drop=True)
    out["score"] = scores
    out["prediction"] = (scores >= threshold).astype(int)
    out["outcome"] = np.where((out.label == 0) & (out.prediction == 1), "false_positive", np.where((out.label == 1) & (out.prediction == 0), "false_negative", "correct"))
    ix = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    for name in ("url_length", "hostname_length", "subdomain_length", "digit_count", "hyphen_count", "slash_count", "is_ip_host", "is_url_shortener", "suspicious_token_count", "is_https", "path_length", "has_query"):
        out[name] = X[:, ix[name]]
    parsed = out.url.map(lambda value: urllib.parse.urlparse(value if "://" in value else "http://" + value))
    out["has_fragment"] = parsed.map(lambda p: int(bool(p.fragment)))
    out["has_nonempty_path"] = (out.path_length > 0).astype(int)
    return out


def subgroup_summary(errors: pd.DataFrame) -> pd.DataFrame:
    fields = ["source", "is_https", "has_nonempty_path", "has_query", "has_fragment", "is_ip_host", "is_url_shortener"]
    rows: list[dict] = []
    for field in fields:
        for value, part in errors.groupby(field, dropna=False, sort=True):
            legit = part[part.label == 0]
            phish = part[part.label == 1]
            rows.append({"group": field, "value": str(value), "rows": int(len(part)), "legitimate": int(len(legit)), "phishing": int(len(phish)), "false_positives": int((legit.outcome == "false_positive").sum()), "false_positive_rate": float((legit.outcome == "false_positive").mean()) if len(legit) else None, "false_negatives": int((phish.outcome == "false_negative").sum()), "false_negative_rate": float((phish.outcome == "false_negative").mean()) if len(phish) else None})
    # Continuous structural attributes are summarized by actual/predicted class.
    for field in ("url_length", "hostname_length", "subdomain_length", "digit_count", "hyphen_count", "slash_count", "suspicious_token_count"):
        for outcome, part in errors.groupby("outcome", sort=True):
            rows.append({"group": field + "_by_outcome", "value": outcome, "rows": int(len(part)), "mean": float(part[field].mean()), "median": float(part[field].median()), "p95": float(part[field].quantile(.95))})
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Offline PSL makes extraction repeatable and prohibits suffix-list requests.
    url_features.tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=str(ROOT / "reports" / "p6" / ".tldextract"))
    train_path, val_path = DATA / "p7_url_train.csv", DATA / "p7_url_val.csv"
    train, val = pd.read_csv(train_path), pd.read_csv(val_path)
    if set(train.source) - {"PHIUSIL", "MURL"} or set(val.source) - {"PHIUSIL", "MURL"}:
        raise RuntimeError("P8 boundary violation: P7 development partitions contain an unapproved source")
    fit = fit_sample(train)
    cv = deterministic_sample(fit, CV_ROWS)
    X_fit = urls_to_feature_matrix(fit.url.astype(str).tolist())
    X_val = urls_to_feature_matrix(val.url.astype(str).tolist())
    X_cv = urls_to_feature_matrix(cv.url.astype(str).tolist())
    y_fit, y_val, y_cv = fit.label.to_numpy(int), val.label.to_numpy(int), cv.label.to_numpy(int)
    groups = component_groups(cv)
    group_counts = pd.Series(groups).value_counts()
    gkf = GroupKFold(n_splits=CV_FOLDS)

    # Verify (rather than overwrite) P7 artifacts, and reproduce their recorded
    # validation metrics in a new P8 result file.
    p7: dict[str, dict] = {}
    for stem, name in (("logisticregression", "LogisticRegression"), ("randomforest", "RandomForest")):
        artifact, metadata = _load_artifact(MODELS / f"p7_url_{stem}.joblib", MODELS / f"p7_url_{stem}_metadata.json")
        score = artifact.predict_proba(X_val)[:, 1]
        result = metric_bundle(y_val, score, float(metadata["threshold"]))
        p7[name] = {"artifact_sha256": sha256(MODELS / f"p7_url_{stem}.joblib"), "metadata_sha256": sha256(MODELS / f"p7_url_{stem}_metadata.json"), "threshold": float(metadata["threshold"]), "reproduced_metrics": result}

    variants: dict[str, dict] = {}
    # Exploratory grouped CV uses only 60k deterministic training-partition rows
    # and the fixed 0.5 operating point.  It ranks feature sets without tuning a
    # decision threshold on every fold.
    for variant in GROUPS:
        names, idx = subset_names(variant), feature_indices(subset_names(variant))
        fold_results = []
        for fold, (learn, held) in enumerate(gkf.split(X_cv, y_cv, groups), start=1):
            model = CANDIDATE_PIPELINES["LogisticRegression"]()
            model.fit(X_cv[learn][:, idx], y_cv[learn])
            score = model.predict_proba(X_cv[held][:, idx])[:, 1]
            fold_results.append({"fold": fold, **metric_bundle(y_cv[held], score, .5)})
        cv_mean = {key: float(np.mean([row[key] for row in fold_results])) for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate_legitimate", "false_negative_rate_phishing")}
        # Reused P7 validation is deliberately shown as a separate, non-final
        # comparison.  Threshold is max-F1 exactly as P7, with no target FPR.
        model = CANDIDATE_PIPELINES["LogisticRegression"]()
        model.fit(X_fit[:, idx], y_fit)
        threshold = float(select_threshold(model, X_val[:, idx], y_val))
        score = model.predict_proba(X_val[:, idx])[:, 1]
        variants[variant] = {"feature_names": names, "removed_features": list(GROUPS[variant]), "grouped_cv": {"folds": fold_results, "mean": cv_mean, "f1_std": float(np.std([r["f1"] for r in fold_results], ddof=1)), "roc_auc_std": float(np.std([r["roc_auc"] for r in fold_results], ddof=1))}, "p7_validation_reuse": {"threshold": threshold, **metric_bundle(y_val, score, threshold)}}

    # Predeclared selection: highest mean grouped-CV F1; ties are broken by lower
    # grouped-CV FPR then canonical feature count.  This avoids selecting from
    # the repeatedly reused P7 validation partition.
    selected = sorted(variants, key=lambda name: (-variants[name]["grouped_cv"]["mean"]["f1"], variants[name]["grouped_cv"]["mean"]["false_positive_rate_legitimate"], -len(variants[name]["feature_names"]), name))[0]
    selected_names, selected_idx = subset_names(selected), feature_indices(subset_names(selected))
    candidates: dict[str, dict] = {}
    for name, factory in CANDIDATE_PIPELINES.items():
        model = factory(); model.fit(X_fit[:, selected_idx], y_fit)
        threshold = float(select_threshold(model, X_val[:, selected_idx], y_val))
        score = model.predict_proba(X_val[:, selected_idx])[:, 1]
        stem = name.lower()
        artifact_path = MODELS / f"p8_url_{stem}_{selected}.joblib"
        metadata_path = MODELS / f"p8_url_{stem}_{selected}_metadata.json"
        metadata = _stamp_schema({"model_name": name, "model_variant": "p8_candidate", "feature_subset": selected, "feature_names": selected_names, "threshold": threshold, "label_map": {"0": "legitimate", "1": "phishing"}, "random_seed": SEED, "purpose": "P8 development-only candidate; not served or promoted", "training_data": "P7 deterministic 150,000-row fit sample (PHIUSIL + conflict-quarantined MURL)", "validation_data": "P7 validation partition; previously used development data", "threshold_selection": "max validation F1; no target false-positive rate", "validation_metrics": metric_bundle(y_val, score, threshold), "created_at_utc": datetime.now(timezone.utc).isoformat()})
        joblib.dump(model, artifact_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _load_artifact(artifact_path, metadata_path)
        candidates[name] = {"artifact": str(artifact_path.relative_to(ROOT)), "artifact_sha256": sha256(artifact_path), "metadata": str(metadata_path.relative_to(ROOT)), "metadata_sha256": sha256(metadata_path), "threshold": threshold, **metric_bundle(y_val, score, threshold)}

    # Diagnose the reproduced P7 Random Forest so error root causes are tied to
    # the baseline, not inferred from an ablation alone.
    baseline, baseline_meta = _load_artifact(MODELS / "p7_url_randomforest.joblib", MODELS / "p7_url_randomforest_metadata.json")
    errors = structural_frame(val, X_val, baseline.predict_proba(X_val)[:, 1], float(baseline_meta["threshold"]))
    subgroup_summary(errors).to_csv(REPORTS / "p8_error_subgroups.csv", index=False)
    errors[errors.outcome != "correct"].sort_values(["outcome", "score", "url"], kind="stable").groupby("outcome", group_keys=False).head(25).to_csv(REPORTS / "p8_representative_errors.csv", index=False)

    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": {"development_sources": ["PHIUSIL", "MURL"], "frozen_comp_phish_read": False, "network_or_destination_access": False, "new_features": False, "canonical_schema": list(FEATURE_NAMES)}, "inputs": {"p7_train_sha256": sha256(train_path), "p7_val_sha256": sha256(val_path), "fit_rows": int(len(fit)), "cv_rows": int(len(cv)), "cv_components": int(len(group_counts)), "cv_largest_component_rows": int(group_counts.max()), "cv_p95_component_rows": float(group_counts.quantile(.95)), "cv_protocol": "3-fold GroupKFold on components linked by registered domain or normalized template; fixed threshold 0.5"}, "p7_reproduction": p7, "variants": variants, "selected_variant": {"name": selected, "rule": "highest grouped-CV mean F1; then lower grouped-CV FPR; then more retained canonical features", "feature_names": selected_names}, "p8_candidates": candidates, "diagnostic_outputs": {"subgroups": "reports/p8/p8_error_subgroups.csv", "representative_errors": "reports/p8/p8_representative_errors.csv", "baseline": "P7 RandomForest reproduced on P7 validation"}, "limitations": ["The P7 validation partition was previously used for candidate and threshold selection and is reported only as reused development evidence.", "Grouped CV is exploratory and runs on a deterministic 60,000-row subset of the training partition for tractability.", "PHIUSIL/MURL structural and source biases remain; no P8 result establishes external generalization."]}
    (REPORTS / "p8_experiments.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_variant": selected, "p8_candidates": candidates, "report": str((REPORTS / "p8_experiments.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
