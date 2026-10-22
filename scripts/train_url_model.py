"""
train_url_model.py
==================
Trains and selects the best URL phishing detection pipeline.

Usage (from repo root):
    python scripts/train_url_model.py

Reads:
    data/processed/urls_train.csv
    data/processed/urls_val.csv
    data/processed/urls_template_train.csv   (domain-isolated pool)

Writes:
    models/url_model.joblib
    models/url_model_metadata.json
    models/url_model_domain_holdout.joblib
    models/url_model_domain_holdout_metadata.json
    reports/url_metrics.json
    reports/model_comparison.csv (rows for the url branch replaced)
"""

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.url_model import (
    CANDIDATE_PIPELINES,
    FEATURE_SCHEMA_VERSION,
    UrlEvalResult,
    evaluate_pipeline,
    save_domain_holdout_model,
    save_model,
    save_revised_model,
    select_threshold,
    urls_to_feature_matrix,
)
from fraud_phishing.features.url_features import FEATURE_NAMES
from fraud_phishing.features.feature_subsets import (
    REVISED_FEATURES,
    REVISED_MODEL_KIND,
    REVISED_SUBSET_NAME,
)
from fraud_phishing.reporting import update_model_comparison

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# For large datasets, subsample to keep training time reasonable
MAX_TRAIN_SAMPLES = 150_000


def load_urls_csv(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/fraud_phishing/data/prepare_urls.py"
        )
    df = pd.read_csv(path)
    return df["url"].astype(str).tolist(), df["label"].astype(int).values


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    return load_urls_csv(PROCESSED / f"urls_{name}.csv")


def train_domain_holdout_model(model_name: str, leakage: dict | None) -> None:
    """
    Fit a model on the domain-isolated pool and save it for honest evaluation.

    The production model is trained on a *random* split of PHIUSIL, which shares
    most of the domain-aware test URLs (see reports/url_leakage_report.json).
    Evaluating it on the domain test set therefore measures memorisation, not
    generalisation to unseen domains. This function trains a separate model on
    data/processed/urls_template_train.csv -- the domain-isolated pool -- and
    selects its threshold on a domain-held-out slice, so neither test set is
    ever used for model selection.
    """
    print(f"\n{'='*60}")
    print("Domain-holdout model (for unseen-domain evaluation)")
    print(f"{'='*60}")

    pool_path = PROCESSED / "urls_template_train.csv"
    if not pool_path.exists():
        print(
            f"[WARN] {pool_path} not found - skipping domain-holdout model. "
            "Run src/fraud_phishing/data/prepare_urls.py first."
        )
        return

    urls, y = load_urls_csv(pool_path)
    print(f"Pool: {len(urls):,} URLs from {pool_path.name}")

    t0 = time.perf_counter()
    X = urls_to_feature_matrix(urls)
    print(f"  Features extracted in {time.perf_counter()-t0:.1f}s -> {X.shape}")

    import tldextract  # noqa: PLC0415
    from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

    groups = np.array(
        [
            tldextract.extract(u).top_domain_under_public_suffix or "__none__"
            for u in urls
        ]
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    fit_idx, val_idx = next(gss.split(X, y, groups=groups))
    if set(groups[fit_idx]) & set(groups[val_idx]):
        raise RuntimeError(
            "Domain-held-out validation slice shares a domain with the fitting "
            "slice; the threshold would not be measured on unseen domains."
        )

    probe = CANDIDATE_PIPELINES[model_name]()
    t0 = time.perf_counter()
    probe.fit(X[fit_idx], y[fit_idx])
    print(
        f"  Threshold probe fit in {time.perf_counter()-t0:.2f}s "
        f"({len(fit_idx):,} rows)"
    )
    threshold = select_threshold(probe, X[val_idx], y[val_idx])
    print(
        f"  Threshold selected on {len(val_idx):,} rows from "
        f"{len(set(groups[val_idx])):,} unseen domains: {threshold:.4f}"
    )

    print(f"  Re-fitting {model_name} on the full domain pool ...")
    t0 = time.perf_counter()
    final = CANDIDATE_PIPELINES[model_name]()
    final.fit(X, y)
    print(f"  Fit in {time.perf_counter()-t0:.2f}s")

    metadata = {
        "model_name": model_name,
        "threshold": threshold,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "random_seed": RANDOM_SEED,
        "purpose": (
            "Trained on the domain-isolated pool (urls_template_train.csv) so it "
            "can be evaluated on unseen registered domains. This is NOT the model "
            "served by the API."
        ),
        "train_rows": int(len(y)),
        "threshold_selection": {
            "method": (
                "GroupShuffleSplit on registered domain; threshold chosen on the "
                "held-out 15% of domains"
            ),
            "val_rows": int(len(val_idx)),
            "unseen_domains_in_val": int(len(set(groups[val_idx]))),
        },
        "leakage_report": leakage,
        "training_date": __import__("datetime").datetime.now().isoformat(),
    }
    save_domain_holdout_model(final, metadata)


def train_revised_model() -> dict | None:
    """
    Train the revised URL model on the augmented pool with the subset of
    features selected by the ablation experiment.

    Training data is PHIUSIL + MURL (never LegitPhish or URL-Phish). The
    threshold is selected on a domain-held-out slice of the same pool, so it is
    never tuned on any benchmark or external set.
    """
    print(f"\n{'='*60}")
    print("Revised model (augmented pool + selected feature subset)")
    print(f"{'='*60}")

    train_path = PROCESSED / "urls_aug_train.csv"
    val_path = PROCESSED / "urls_aug_val.csv"
    if not train_path.exists() or not val_path.exists():
        print(
            f"[WARN] {train_path.name} / {val_path.name} not found - skipping "
            "revised model. Run prepare_urls_augmented.py first."
        )
        return None

    urls_train, y_train = load_urls_csv(train_path)
    urls_val, y_val = load_urls_csv(val_path)
    feats = list(REVISED_FEATURES)
    print(f"Feature subset: {REVISED_SUBSET_NAME} ({len(feats)} features)")

    print(f"Extracting features for {len(urls_train):,} training URLs ...")
    X_train = urls_to_feature_matrix(urls_train, feature_names=feats)
    X_val = urls_to_feature_matrix(urls_val, feature_names=feats)
    print(f"Train: {X_train.shape} ({y_train.mean():.3f} positive)  "
          f"Val: {X_val.shape} ({y_val.mean():.3f} positive)")

    results: dict[str, UrlEvalResult] = {}
    fitted: dict = {}
    thresholds: dict[str, float] = {}
    for model_name, build_fn in CANDIDATE_PIPELINES.items():
        print(f"\n-- {model_name} --")
        pipe = build_fn()
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        print(f"  Train time: {time.perf_counter()-t0:.1f}s")
        thr = select_threshold(pipe, X_val, y_val)
        thresholds[model_name] = thr
        res = evaluate_pipeline(pipe, X_val, y_val, model_name, threshold=thr)
        results[model_name] = res
        fitted[model_name] = pipe
        print(f"  Val F1={res.f1:.4f}  ROC-AUC={res.roc_auc:.4f}")

    # The model *family* is not chosen by validation F1. Validation F1 is
    # measured on the same mixture the training data came from, so it rewards the
    # tree model that memorised the "https AND path -> phishing" conjunction
    # (val F1 is higher for RandomForest while its false-positive rate on
    # independent legitimate URLs is ~98%). The family is therefore taken from
    # the ablation, which measured that false-positive rate directly:
    # reports/url_feature_ablation.md.
    if REVISED_MODEL_KIND not in results:
        raise RuntimeError(
            f"REVISED_MODEL_KIND={REVISED_MODEL_KIND!r} is not one of the "
            f"candidate pipelines {sorted(results)}."
        )
    best_name = REVISED_MODEL_KIND
    best_result = results[best_name]
    best_threshold = thresholds[best_name]
    print(
        f"\nSelected revised model: {best_name} (val F1={best_result.f1:.4f}); "
        "family fixed by the ablation experiment, not by validation F1."
    )

    all_urls = urls_train + urls_val
    all_y = np.concatenate([y_train, y_val])
    print(f"Re-fitting {best_name} on train+val ({len(all_urls):,}) ...")
    all_X = urls_to_feature_matrix(all_urls, feature_names=feats)
    final = CANDIDATE_PIPELINES[best_name]()
    final.fit(all_X, all_y)

    aug_stats_path = PROCESSED / "urls_aug_split_stats.json"
    metadata = {
        "model_name": best_name,
        "model_variant": "revised",
        "threshold": best_threshold,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "feature_names": feats,
        "n_features": len(feats),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_subset": REVISED_SUBSET_NAME,
        "feature_subset_rationale": (
            "In PHIUSIL every legitimate URL is a bare https://domain with no path "
            "while phishing URLs carry paths, and the only path-bearing legitimate "
            "source used for augmentation (MURL) stores no scheme, so the "
            "conjunction 'https AND non-empty path' occurred only in the phishing "
            "class. Features that encode URL shape (path, query, length) were "
            "therefore removed, chosen by scripts/ablate_url_features.py "
            "(reports/url_feature_ablation.md)."
        ),
        "model_selection": (
            "Pipeline family fixed to "
            f"{REVISED_MODEL_KIND} by the ablation experiment; validation F1 alone "
            "would have selected RandomForest, which is not used because its "
            "false-positive rate on independent legitimate URLs is ~98%."
        ),
        "random_seed": RANDOM_SEED,
        "trained_on": (
            "data/processed/urls_aug_train.csv (+val for the final fit): PHIUSIL "
            "domain-isolated pool + MURL benign/legitimate/phishing"
        ),
        "never_trained_on": [
            "LegitPhish (Dataset C, development benchmark)",
            "URL-Phish (Dataset E, untouched final external set)",
        ],
        "val_metrics": {
            "f1": best_result.f1,
            "roc_auc": best_result.roc_auc,
            "confusion_matrix": best_result.confusion,
        },
        "all_val_results": {
            n: {"f1": r.f1, "roc_auc": r.roc_auc} for n, r in results.items()
        },
        "threshold_selection": (
            "domain-held-out 15% slice of the augmented pool (urls_aug_val.csv)"
        ),
        "augmented_pool_stats": (
            json.loads(aug_stats_path.read_text()) if aug_stats_path.exists() else {}
        ),
        "training_date": __import__("datetime").datetime.now().isoformat(),
    }
    save_revised_model(final, metadata)

    comp_path = REPORTS / "model_comparison.csv"
    update_model_comparison(
        comp_path,
        "url_revised",
        [
            {
                "branch": "url_revised",
                "model": name,
                "val_f1": f"{res.f1:.4f}",
                "val_roc_auc": f"{res.roc_auc:.4f}",
                "threshold": f"{thresholds[name]:.4f}",
                "selected": name == best_name,
            }
            for name, res in results.items()
        ],
    )
    return metadata


def main() -> None:
    print("=" * 60)
    print("Training URL phishing detection models")
    print("=" * 60)

    urls_train, y_train = load_split("train")
    urls_val, y_val = load_split("val")

    # Subsample large training sets for tractability
    if len(urls_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(urls_train), MAX_TRAIN_SAMPLES, replace=False)
        urls_train = [urls_train[i] for i in idx]
        y_train = y_train[idx]
        print(f"Subsampled train to {MAX_TRAIN_SAMPLES:,} examples.")

    print(f"\nExtracting features for {len(urls_train):,} training URLs …")
    t0 = time.perf_counter()
    X_train = urls_to_feature_matrix(urls_train)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    print(f"Extracting features for {len(urls_val):,} validation URLs …")
    t0 = time.perf_counter()
    X_val = urls_to_feature_matrix(urls_val)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    print(f"\nTrain: {X_train.shape} ({y_train.mean():.3f} positive)")
    print(f"Val  : {X_val.shape} ({y_val.mean():.3f} positive)")

    results: dict[str, UrlEvalResult] = {}
    fitted_pipelines: dict = {}
    thresholds: dict[str, float] = {}

    for model_name, build_fn in CANDIDATE_PIPELINES.items():
        print(f"\n── {model_name} ────────────────────────────────────")
        pipeline = build_fn()

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        print(f"  Train time: {train_time:.2f}s")

        thresh = select_threshold(pipeline, X_val, y_val)
        thresholds[model_name] = thresh

        result = evaluate_pipeline(pipeline, X_val, y_val, model_name, threshold=thresh)
        results[model_name] = result
        fitted_pipelines[model_name] = pipeline

        print(f"  Val F1={result.f1:.4f}  ROC-AUC={result.roc_auc:.4f}")
        print(result.report)

    # ── Select best model ──────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n].f1)
    best_result = results[best_name]
    best_pipeline = fitted_pipelines[best_name]
    best_threshold = thresholds[best_name]

    print(f"\n{'='*60}")
    print(f"Selected URL model: {best_name}  (val F1={best_result.f1:.4f})")
    print(f"{'='*60}")

    # Re-fit on train+val
    all_urls = urls_train + urls_val
    all_y = np.concatenate([y_train, y_val])
    print(f"\nRe-fitting {best_name} on train+val ({len(all_urls):,} examples) …")
    all_X = urls_to_feature_matrix(all_urls)
    best_final = CANDIDATE_PIPELINES[best_name]()
    best_final.fit(all_X, all_y)

    stats_path = PROCESSED / "urls_split_stats.json"
    split_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    metadata = {
        "model_name": best_name,
        "threshold": best_threshold,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "random_seed": RANDOM_SEED,
        "split_stats": split_stats,
        "val_metrics": {
            "f1": best_result.f1,
            "roc_auc": best_result.roc_auc,
            "confusion_matrix": best_result.confusion,
        },
        "all_val_results": {
            name: {"f1": r.f1, "roc_auc": r.roc_auc}
            for name, r in results.items()
        },
        "training_date": __import__("datetime").datetime.now().isoformat(),
    }
    save_model(best_final, metadata)

    # ── Domain-isolated model for honest unseen-domain evaluation ─────────
    train_domain_holdout_model(best_name, split_stats.get("leakage"))

    # ── Revised model: augmented pool + ablated feature subset ────────────
    train_revised_model()

    metrics_out = {
        "split": "validation",
        "model": best_name,
        "threshold": best_threshold,
        **metadata["val_metrics"],
    }
    (REPORTS / "url_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )

    comp_path = REPORTS / "model_comparison.csv"
    update_model_comparison(
        comp_path,
        "url",
        [
            {
                "branch": "url",
                "model": name,
                "val_f1": f"{res.f1:.4f}",
                "val_roc_auc": f"{res.roc_auc:.4f}",
                "threshold": f"{thresholds[name]:.4f}",
                "selected": name == best_name,
            }
            for name, res in results.items()
        ],
    )
    print(f"\nModel comparison: {comp_path}")
    print(f"URL metrics     : {REPORTS / 'url_metrics.json'}")
    print("\nDone. Run evaluate_url_model.py for test-set results.")


if __name__ == "__main__":
    main()
