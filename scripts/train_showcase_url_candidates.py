"""Train, compare, and conditionally integrate URL models for the college showcase.

This is a development-only model-selection path. It trains Logistic Regression
and Random Forest candidates on the grouped showcase training partition using
the existing static extractor.  It compares the P7 full schema and the P8
``without_brand`` subset, chooses an operating threshold on validation only,
and scores the selected candidate once on the untouched grouped test partition.

If predeclared internal acceptance criteria pass, the selected configuration is
refit on train+validation at its already-fixed threshold and copied into the
served URL artifact after preserving a byte-for-byte rollback copy.  SMS files,
P9 artifacts, the frozen CompPhish CSV, and historical candidate artifacts are
never changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: E402
from fraud_phishing.models.url_model import (  # noqa: E402
    CANDIDATE_PIPELINES,
    URL_METADATA_PATH,
    URL_MODEL_PATH,
    _load_artifact,
    _stamp_schema,
    feature_indices,
    predict_url,
    select_threshold,
    urls_to_feature_matrix,
)

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "showcase"
MODELS = ROOT / "models"
SEED = 42
MAX_FIT_ROWS = 150_000
VARIANTS = {
    "p7_all_features": list(FEATURE_NAMES),
    "p8_without_brand": [name for name in FEATURE_NAMES if name not in {"brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain"}],
}
# These criteria are declared before results are calculated. They are a modest
# integration gate for a college demonstration, not a claim of production
# security fitness or external validity.
ACCEPTANCE = {
    "minimum_test_f1": 0.80,
    "minimum_test_roc_auc": 0.85,
    "maximum_test_false_positive_rate": 0.20,
    "maximum_test_false_negative_rate": 0.20,
    "minimum_test_path_or_query_legitimate_fpr": None,
    "minimum_f1_improvement_over_served": 0.05,
    "minimum_fpr_reduction_over_served": 0.05,
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False, encoding="utf-8") as handle:
        temp = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2)
    try:
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_joblib(model: object, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as handle:
        temp = pathlib.Path(handle.name)
    try:
        joblib.dump(model, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def deterministic_fit_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Bound training cost while retaining every available CompPhish train row."""
    if len(frame) <= MAX_FIT_ROWS:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    selected: list[pd.DataFrame] = []
    remaining_by_label: dict[int, int] = {}
    for label in (0, 1):
        comp = frame[(frame.source == "CompPhish") & (frame.label == label)].sort_values("url", kind="stable")
        selected.append(comp)
        remaining_by_label[label] = max((MAX_FIT_ROWS // 2) - len(comp), 0)
    for label in (0, 1):
        eligible = frame[(frame.source != "CompPhish") & (frame.label == label)].sort_values("url", kind="stable")
        target = min(remaining_by_label[label], len(eligible))
        if target:
            indices = np.sort(rng.choice(len(eligible), target, replace=False))
            selected.append(eligible.iloc[indices])
    return pd.concat(selected, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate_legitimate": float(fp / max(tn + fp, 1)),
        "false_negative_rate_phishing": float(fn / max(fn + tp, 1)),
        "predicted_class_counts": {"legitimate": int((pred == 0).sum()), "phishing": int((pred == 1).sum())},
        "rows": int(len(y)),
        "legitimate": int((y == 0).sum()),
        "phishing": int((y == 1).sum()),
    }


def legitimate_structure_metrics(frame: pd.DataFrame, X_full: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    fields = {name: FEATURE_NAMES.index(name) for name in ("path_length", "has_query")}
    legit = frame.label.to_numpy(dtype=int) == 0
    groups = {
        "nonempty_path": X_full[:, fields["path_length"]] > 0,
        "query_parameter": X_full[:, fields["has_query"]] == 1,
        "path_or_query": (X_full[:, fields["path_length"]] > 0) | (X_full[:, fields["has_query"]] == 1),
    }
    prediction = (scores >= threshold).astype(int)
    report: dict[str, dict] = {}
    for name, mask in groups.items():
        mask = mask & legit
        total = int(mask.sum())
        false_positive = int((prediction[mask] == 1).sum())
        report[name] = {
            "legitimate_rows": total,
            "false_positives": false_positive,
            "false_positive_rate": None if total == 0 else float(false_positive / total),
        }
    return report


def candidate_entry(
    name: str,
    variant: str,
    model: object,
    threshold: float,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_frame: pd.DataFrame,
    X_test_full: np.ndarray,
) -> dict:
    validation_scores = model.predict_proba(X_validation)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return {
        "model_name": name,
        "feature_configuration": variant,
        "threshold": threshold,
        "threshold_selection": "maximum F1 on grouped validation partition",
        "validation": metrics(y_validation, validation_scores, threshold),
        "test": {
            **metrics(y_test, test_scores, threshold),
            "legitimate_path_query_subgroups": legitimate_structure_metrics(test_frame, X_test_full, test_scores, threshold),
        },
    }


def validation_rank(entry: dict) -> tuple:
    value = entry["validation"]
    return (-value["f1"], value["false_positive_rate_legitimate"], value["false_negative_rate_phishing"], entry["model_name"])


def meets_acceptance(selected: dict, served: dict) -> tuple[bool, list[str]]:
    candidate, baseline = selected["test"], served["test"]
    checks = [
        (candidate["f1"] >= ACCEPTANCE["minimum_test_f1"], f"test F1 {candidate['f1']:.4f} >= {ACCEPTANCE['minimum_test_f1']:.2f}"),
        (candidate["roc_auc"] >= ACCEPTANCE["minimum_test_roc_auc"], f"test ROC-AUC {candidate['roc_auc']:.4f} >= {ACCEPTANCE['minimum_test_roc_auc']:.2f}"),
        (candidate["false_positive_rate_legitimate"] <= ACCEPTANCE["maximum_test_false_positive_rate"], f"test FPR {candidate['false_positive_rate_legitimate']:.4f} <= {ACCEPTANCE['maximum_test_false_positive_rate']:.2f}"),
        (candidate["false_negative_rate_phishing"] <= ACCEPTANCE["maximum_test_false_negative_rate"], f"test FNR {candidate['false_negative_rate_phishing']:.4f} <= {ACCEPTANCE['maximum_test_false_negative_rate']:.2f}"),
        (candidate["f1"] - baseline["f1"] >= ACCEPTANCE["minimum_f1_improvement_over_served"], f"test F1 improvement {candidate['f1'] - baseline['f1']:.4f} >= {ACCEPTANCE['minimum_f1_improvement_over_served']:.2f}"),
        (baseline["false_positive_rate_legitimate"] - candidate["false_positive_rate_legitimate"] >= ACCEPTANCE["minimum_fpr_reduction_over_served"], f"test FPR reduction {baseline['false_positive_rate_legitimate'] - candidate['false_positive_rate_legitimate']:.4f} >= {ACCEPTANCE['minimum_fpr_reduction_over_served']:.2f}"),
    ]
    return all(ok for ok, _ in checks), [f"{'PASS' if ok else 'FAIL'}: {text}" for ok, text in checks]


def write_markdown(report: dict, destination: pathlib.Path) -> None:
    selected = report["selection"]["selected_candidate"]
    lines = [
        "# Final Showcase URL Classifier",
        "",
        "## Scope",
        "",
        "This controlled development run uses PHIUSIIL, conflict-screened MURL, and a byte-identical development copy of the previously evaluated CompPhish subset. The original frozen CSV, P6 manifest, and P9 result remain unchanged. CompPhish is development data in this run and is not an independent test.",
        "",
        "All features are static lexical features extracted from raw URL strings. No HTML-derived fields, DNS, HTTP requests, or URL visits are used.",
        "",
        "## Split integrity",
        "",
        f"The deterministic seed is {report['split_manifest']['splits']['seed']}. Train, validation, and test groups are connected components formed by registered domain and normalized URL template. Every pairwise overlap count for raw URL, normalized URL, registered domain, and template is zero; see `showcase_split_manifest.json`.",
        "",
        "## Candidates",
        "",
        "Thresholds maximize F1 on validation only. The grouped test partition is scored once after the validation-based selection.",
        "",
        "| Candidate | Features | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test FPR | Test FNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, candidate in report["candidates"].items():
        validation, test = candidate["validation"], candidate["test"]
        lines.append(f"| {key} | {candidate['feature_configuration']} | {validation['f1']:.4f} | {test['f1']:.4f} | {test['roc_auc']:.4f} | {test['pr_auc']:.4f} | {test['false_positive_rate_legitimate']:.2%} | {test['false_negative_rate_phishing']:.2%} |")
    lines += [
        "",
        "## Selected candidate and integration gate",
        "",
        f"Validation selection chose **{selected}** using highest validation F1, then lower FPR and FNR. The served baseline was evaluated on the same grouped test partition for comparison only.",
        "",
        *[f"- {line}" for line in report["selection"]["acceptance_checks"]],
        "",
        f"**Integrated:** {report['integration']['integrated']}. {report['integration']['reason']}",
        "",
        "## Legitimate URLs with paths and queries",
        "",
        "| Subgroup | Legitimate rows | False positives | FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, subgroup in report["candidates"][selected]["test"]["legitimate_path_query_subgroups"].items():
        rate = "n/a" if subgroup["false_positive_rate"] is None else f"{subgroup['false_positive_rate']:.2%}"
        lines.append(f"| {name} | {subgroup['legitimate_rows']} | {subgroup['false_positives']} | {rate} |")
    lines += [
        "",
        "## PhishVN diagnostic",
        "",
        "PhishVN v3.1.0 is verified against its internal manifest and is accepted as the official v4 documentation-only-equivalent archive. It is **not scored here**: the raw URL field is scheme-stripped, and treating missing schemes as HTTP would invent a feature value for `is_https`. Its earlier structural analysis and verified-core/bronze separation are retained in `reports/p11/p11_phishvn_archive_qualification.json`.",
        "",
        "## Limits",
        "",
        "This is grouped internal development evidence, not a new independent benchmark. P9's historical external failure remains valid for its P7 model and cannot be compared directly with the new candidate. The URL classifier remains a college-project prototype rather than a production security control.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads((REPORTS / "showcase_split_manifest.json").read_text(encoding="utf-8"))
    if split_manifest.get("status") != "complete" or any(value for pair in split_manifest["overlap"].values() for value in pair.values()):
        raise RuntimeError("Showcase split manifest is missing or does not prove zero prohibited overlap")
    train = pd.read_csv(DATA / "showcase_url_train.csv")
    validation = pd.read_csv(DATA / "showcase_url_validation.csv")
    test = pd.read_csv(DATA / "showcase_url_test.csv")
    if set(pd.concat([train.source, validation.source, test.source])) != {"PHIUSIL", "MURL", "CompPhish"}:
        raise RuntimeError("Showcase partitions do not contain the approved development sources")
    fit = deterministic_fit_sample(train)
    X_fit_full = urls_to_feature_matrix(fit.url.astype(str).tolist())
    X_validation_full = urls_to_feature_matrix(validation.url.astype(str).tolist())
    X_test_full = urls_to_feature_matrix(test.url.astype(str).tolist())
    y_fit = fit.label.to_numpy(dtype=int)
    y_validation = validation.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)

    candidates: dict[str, dict] = {}
    trained: dict[str, tuple[object, list[str]]] = {}
    for variant, names in VARIANTS.items():
        indices = feature_indices(names)
        for model_name, factory in CANDIDATE_PIPELINES.items():
            model = factory()
            model.fit(X_fit_full[:, indices], y_fit)
            if list(model.classes_) != [0, 1]:
                raise RuntimeError(f"{variant}/{model_name} class order is {model.classes_}")
            threshold = float(select_threshold(model, X_validation_full[:, indices], y_validation))
            key = f"{model_name}_{variant}"
            entry = candidate_entry(model_name, variant, model, threshold, X_validation_full[:, indices], y_validation, X_test_full[:, indices], y_test, test, X_test_full)
            artifact = MODELS / f"showcase_url_{model_name.lower()}_{variant}.joblib"
            metadata = MODELS / f"showcase_url_{model_name.lower()}_{variant}_metadata.json"
            metadata_payload = _stamp_schema({
                "model_name": model_name,
                "model_variant": "showcase_candidate",
                "feature_configuration": variant,
                **({"feature_subset": variant} if variant != "p7_all_features" else {}),
                "feature_names": names,
                "threshold": threshold,
                "label_map": {"0": "legitimate", "1": "phishing"},
                "random_seed": SEED,
                "purpose": "Final-showcase candidate; selection on validation and one grouped internal test score",
                "training_data": "showcase_url_train.csv deterministic source-aware fit sample",
                "validation_data": "showcase_url_validation.csv connected domain/template held-out",
                "test_data": "showcase_url_test.csv connected domain/template held-out; scored after validation selection",
                "threshold_selection": "maximum validation F1",
                "validation_metrics": entry["validation"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            atomic_joblib(model, artifact)
            atomic_write_json(metadata_payload, metadata)
            _load_artifact(artifact, metadata)
            entry.update({"artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": sha256(artifact), "metadata": str(metadata.relative_to(ROOT)), "metadata_sha256": sha256(metadata)})
            candidates[key] = entry
            trained[key] = (model, names)

    selected_key = sorted(candidates, key=lambda key: validation_rank(candidates[key]))[0]
    served_model, served_metadata = _load_artifact(URL_MODEL_PATH, URL_METADATA_PATH)
    served_names = list(served_metadata["feature_names"])
    served_index = feature_indices(served_names)
    served_threshold = float(served_metadata["threshold"])
    served_scores = served_model.predict_proba(X_test_full[:, served_index])[:, 1]
    served = {
        "artifact": str(URL_MODEL_PATH),
        "artifact_sha256_before": sha256(URL_MODEL_PATH),
        "metadata": str(URL_METADATA_PATH),
        "metadata_sha256_before": sha256(URL_METADATA_PATH),
        "threshold": served_threshold,
        "test": {**metrics(y_test, served_scores, served_threshold), "legitimate_path_query_subgroups": legitimate_structure_metrics(test, X_test_full, served_scores, served_threshold)},
    }
    accepted, acceptance_checks = meets_acceptance(candidates[selected_key], served)

    integration = {"integrated": False, "reason": "Acceptance criteria were not met; served URL model remains unchanged."}
    if accepted:
        selected = candidates[selected_key]
        selected_model_name = selected["model_name"]
        selected_names = selected["feature_configuration"]
        selected_feature_names = VARIANTS[selected_names]
        selected_indices = feature_indices(selected_feature_names)
        # Refit only after the test decision, with validation added to training;
        # retain the validation-derived threshold exactly as selected.
        final_fit = deterministic_fit_sample(pd.concat([train, validation], ignore_index=True))
        final_model = CANDIDATE_PIPELINES[selected_model_name]().fit(
            urls_to_feature_matrix(final_fit.url.astype(str).tolist())[:, selected_indices],
            final_fit.label.to_numpy(dtype=int),
        )
        final_metadata = _stamp_schema({
            "model_name": selected_model_name,
            "model_variant": "showcase_integrated",
            "feature_configuration": selected_names,
            **({"feature_subset": selected_names} if selected_names != "p7_all_features" else {}),
            "feature_names": selected_feature_names,
            "threshold": float(selected["threshold"]),
            "label_map": {"0": "legitimate", "1": "phishing"},
            "random_seed": SEED,
            "purpose": "College showcase served URL model; grouped internal acceptance criteria passed",
            "training_data": "showcase train + validation deterministic source-aware fit sample",
            "threshold_selection": "fixed from pre-refit grouped validation max-F1 selection",
            "selection_test_metrics": selected["test"],
            "acceptance_criteria": ACCEPTANCE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        final_path = MODELS / "url_model_showcase.joblib"
        final_meta_path = MODELS / "url_model_showcase_metadata.json"
        atomic_joblib(final_model, final_path)
        atomic_write_json(final_metadata, final_meta_path)
        _load_artifact(final_path, final_meta_path)
        rollback_model = MODELS / "url_model_pre_showcase.joblib"
        rollback_meta = MODELS / "url_model_pre_showcase_metadata.json"
        # The rollback is a historical snapshot. Once created, retain it across
        # reproducibility reruns instead of replacing it with the then-current
        # served artifact.
        if rollback_model.exists() or rollback_meta.exists():
            if not rollback_model.exists() or not rollback_meta.exists():
                raise RuntimeError("Rollback artifact pair is incomplete; refusing served-model replacement")
        else:
            shutil.copyfile(URL_MODEL_PATH, rollback_model)
            shutil.copyfile(URL_METADATA_PATH, rollback_meta)
            if sha256(rollback_model) != served["artifact_sha256_before"] or sha256(rollback_meta) != served["metadata_sha256_before"]:
                raise RuntimeError("Rollback copy hash mismatch; refusing served-model replacement")
        shutil.copyfile(final_path, URL_MODEL_PATH)
        shutil.copyfile(final_meta_path, URL_METADATA_PATH)
        loaded_model, loaded_meta = _load_artifact(URL_MODEL_PATH, URL_METADATA_PATH)
        probe = predict_url("https://example.org/account?item=1", loaded_model, float(loaded_meta["threshold"]), loaded_meta["feature_names"])
        if probe["label"] != int(probe["score"] >= probe["threshold"]):
            raise RuntimeError("Promoted artifact probability/label mapping failed")
        integration = {
            "integrated": True,
            "reason": "All predeclared grouped internal acceptance criteria passed.",
            "served_artifact": str(URL_MODEL_PATH),
            "served_artifact_sha256": sha256(URL_MODEL_PATH),
            "served_metadata": str(URL_METADATA_PATH),
            "served_metadata_sha256": sha256(URL_METADATA_PATH),
            "showcase_artifact": str(final_path.relative_to(ROOT)),
            "showcase_artifact_sha256": sha256(final_path),
            "rollback_artifact": str(rollback_model.relative_to(ROOT)),
            "rollback_artifact_sha256": sha256(rollback_model),
            "rollback_metadata": str(rollback_meta.relative_to(ROOT)),
            "rollback_metadata_sha256": sha256(rollback_meta),
        }

    report = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"static_raw_url_features_only": True, "html_derived_features_used": False, "destination_access": False, "sms_branch_changed": False, "compphish_role": "development data; P9 historical evidence preserved"},
        "split_manifest": split_manifest,
        "fit_sample": {"rows": int(len(fit)), "class_counts": {str(key): int(value) for key, value in fit.label.value_counts().sort_index().items()}, "source_counts": {str(key): int(value) for key, value in fit.source.value_counts().sort_index().items()}, "rule": "retain CompPhish train rows; fill remaining per-class quota with deterministic PHIUSIIL/MURL samples"},
        "candidate_configurations": {name: {"feature_names": names, "feature_count": len(names)} for name, names in VARIANTS.items()},
        "candidates": candidates,
        "served_baseline": served,
        "selection": {"rule": "highest validation F1; ties lower validation FPR, then lower validation FNR", "selected_candidate": selected_key, "acceptance_criteria": ACCEPTANCE, "acceptance_checks": acceptance_checks},
        "integration": integration,
        "phishvn_diagnostic": {"evaluated": False, "reason": "verified-core URLs are scheme-stripped; using the extractor would synthesize HTTP/is_https=0 and invalidate a scheme-sensitive diagnostic", "verified_archive_report": "reports/p11/p11_phishvn_archive_qualification.json", "prior_structural_analysis_exposure": True},
        "limitations": ["The grouped test partition is internal development data, not a new independent benchmark.", "CompPhish was previously used for P9 and is now repurposed as development data; its historical P9 result remains unchanged.", "PhishVN was not scored because its scheme-stripped URLs cannot provide an honest is_https feature value.", "The integrated URL model remains a college-project prototype and must not be treated as a production security control."],
    }
    atomic_write_json(report, REPORTS / "showcase_url_results.json")
    write_markdown(report, REPORTS / "final_showcase_url_classifier.md")
    print(json.dumps({"selected_candidate": selected_key, "integration": integration, "report": str((REPORTS / "showcase_url_results.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
