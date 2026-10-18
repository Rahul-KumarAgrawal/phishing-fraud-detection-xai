"""
train_url_model.py
==================
Trains and selects the best URL phishing detection pipeline.

Usage (from repo root):
    python scripts/train_url_model.py

Reads:
    data/processed/urls_train.csv
    data/processed/urls_val.csv

Writes:
    models/url_model.joblib
    models/url_model_metadata.json
    reports/url_metrics.json
    reports/model_comparison.csv (appended)
"""

import csv
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.url_model import (
    CANDIDATE_PIPELINES,
    UrlEvalResult,
    evaluate_pipeline,
    save_model,
    select_threshold,
    urls_to_feature_matrix,
)
from fraud_phishing.features.url_features import FEATURE_NAMES

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# For large datasets, subsample to keep training time reasonable
MAX_TRAIN_SAMPLES = 150_000


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    path = PROCESSED / f"urls_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/fraud_phishing/data/prepare_urls.py"
        )
    df = pd.read_csv(path)
    return df["url"].astype(str).tolist(), df["label"].astype(int).values


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
    write_header = not comp_path.exists()
    with open(comp_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["branch", "model", "val_f1", "val_roc_auc", "threshold", "selected"],
        )
        if write_header:
            w.writeheader()
        for name, res in results.items():
            w.writerow(
                {
                    "branch": "url",
                    "model": name,
                    "val_f1": f"{res.f1:.4f}",
                    "val_roc_auc": f"{res.roc_auc:.4f}",
                    "threshold": f"{thresholds[name]:.4f}",
                    "selected": name == best_name,
                }
            )
    print(f"\nModel comparison: {comp_path}")
    print(f"URL metrics     : {REPORTS / 'url_metrics.json'}")
    print("\nDone. Run evaluate_url_model.py for test-set results.")


if __name__ == "__main__":
    main()
