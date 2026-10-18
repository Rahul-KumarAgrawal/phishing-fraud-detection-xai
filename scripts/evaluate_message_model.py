"""
evaluate_message_model.py
=========================
Evaluates the saved message model on the held-out test set.
Run ONCE after training is complete. Do not use test data for tuning.

Usage (from repo root):
    python scripts/evaluate_message_model.py

Reads:
    models/message_model.joblib
    models/message_model_metadata.json
    data/processed/messages_test.csv

Writes:
    reports/message_metrics.json  (updated with test results)
    reports/figures/message_confusion_matrix.png
"""

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.message_model import load_model

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading model …")
    pipeline, metadata = load_model()
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")
    print(f"Model: {model_name}  threshold: {threshold:.4f}")

    test_path = PROCESSED / "messages_test.csv"
    if not test_path.exists():
        print(f"[ERROR] {test_path} not found.")
        sys.exit(1)

    df = pd.read_csv(test_path)
    X_test = df["text"].astype(str)
    y_test = df["label"].astype(int)
    print(f"Test set: {len(X_test):,} ({y_test.mean():.3f} positive)")

    # Inference timing
    t0 = time.perf_counter()
    proba = pipeline.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    ms_per_sample = elapsed / len(X_test) * 1000

    y_pred = (proba >= threshold).astype(int)

    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    report = classification_report(y_test, y_pred, target_names=["ham", "smishing"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n── Test Set Results ──────────────────────────────────────")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Inference : {ms_per_sample:.3f} ms/sample")
    print(f"\n{report}")
    print("Confusion matrix:")
    print(cm)

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "split": "test",
        "model": model_name,
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(X_test)),
        "n_positive_test": int(y_test.sum()),
        "inference_ms_per_sample": ms_per_sample,
        "classification_report": report,
    }
    out_path = REPORTS / "message_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved: {out_path}")

    # ── Confusion matrix figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "smishing"],
        yticklabels=["ham", "smishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Message Model – Confusion Matrix\n{model_name}")
    fig.tight_layout()
    fig_path = FIGURES / "message_confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix figure: {fig_path}")


if __name__ == "__main__":
    main()
