"""
train_message_model.py
======================
Trains and selects the best message (SMS/smishing) classification pipeline.

Usage (from repo root):
    python scripts/train_message_model.py

Reads:
    data/processed/messages_train.csv
    data/processed/messages_val.csv
    data/processed/messages_split_stats.json

Writes:
    models/message_model.joblib
    models/message_model_metadata.json
    reports/message_metrics.json
    reports/model_comparison.csv (appended)
"""

import csv
import json
import pathlib
import sys
import time

import pandas as pd

# Ensure src/ is on path when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.message_model import (
    CANDIDATE_PIPELINES,
    EvalResult,
    MESSAGE_METADATA_PATH,
    MESSAGE_MODEL_PATH,
    evaluate_pipeline,
    save_model,
    select_threshold,
)

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    path = PROCESSED / f"messages_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/fraud_phishing/data/prepare_messages.py"
        )
    df = pd.read_csv(path)
    return df["text"].astype(str), df["label"].astype(int)


def main() -> None:
    print("=" * 60)
    print("Training message classification models")
    print("=" * 60)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print(f"\nTrain: {len(X_train):,} ({y_train.mean():.3f} positive)")
    print(f"Val  : {len(X_val):,} ({y_val.mean():.3f} positive)")

    results: dict[str, EvalResult] = {}
    fitted_pipelines: dict = {}
    thresholds: dict[str, float] = {}

    for model_name, build_fn in CANDIDATE_PIPELINES.items():
        print(f"\n── {model_name} ────────────────────────────────────")
        pipeline = build_fn()

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        print(f"  Train time: {train_time:.2f}s")

        # Select threshold on validation set only
        thresh = select_threshold(pipeline, X_val, y_val)
        thresholds[model_name] = thresh

        # Evaluate on validation set
        result = evaluate_pipeline(pipeline, X_val, y_val, model_name, threshold=thresh)
        results[model_name] = result
        fitted_pipelines[model_name] = pipeline

        print(f"  Val F1={result.f1:.4f}  ROC-AUC={result.roc_auc:.4f}")
        print(result.report)

    # ── Select best model ──────────────────────────────────────────────────
    # Priority: highest F1 on validation set
    best_name = max(results, key=lambda n: results[n].f1)
    best_result = results[best_name]
    best_pipeline = fitted_pipelines[best_name]
    best_threshold = thresholds[best_name]

    print(f"\n{'='*60}")
    print(f"Selected model: {best_name}  (val F1={best_result.f1:.4f})")
    print(f"{'='*60}")

    # ── Save model ────────────────────────────────────────────────────────
    # Re-train on train+val before saving
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    print(f"\nRe-fitting {best_name} on train+val ({len(X_trainval):,} examples) …")
    best_pipeline_final = CANDIDATE_PIPELINES[best_name]()
    best_pipeline_final.fit(X_trainval, y_trainval)

    stats_path = PROCESSED / "messages_split_stats.json"
    split_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    metadata = {
        "model_name": best_name,
        "threshold": best_threshold,
        "label_map": {"0": "ham", "1": "smishing"},
        "feature_type": "TF-IDF word(1-3gram) + char_wb(2-5gram)",
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
    save_model(best_pipeline_final, metadata)

    # ── Write reports ─────────────────────────────────────────────────────
    metrics_out = {
        "split": "validation",
        "model": best_name,
        "threshold": best_threshold,
        **metadata["val_metrics"],
    }
    (REPORTS / "message_metrics.json").write_text(
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
                    "branch": "message",
                    "model": name,
                    "val_f1": f"{res.f1:.4f}",
                    "val_roc_auc": f"{res.roc_auc:.4f}",
                    "threshold": f"{thresholds[name]:.4f}",
                    "selected": name == best_name,
                }
            )
    print(f"\nModel comparison written to: {comp_path}")
    print(f"Message metrics written to : {REPORTS / 'message_metrics.json'}")
    print("\nDone. Run evaluate_message_model.py for test-set results.")


if __name__ == "__main__":
    main()
