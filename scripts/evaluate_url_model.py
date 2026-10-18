"""
evaluate_url_model.py
=====================
Evaluates the saved URL model on the held-out test set (both random and domain-aware).

Usage (from repo root):
    python scripts/evaluate_url_model.py
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
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.url_model import load_model, urls_to_feature_matrix

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def evaluate_split(pipeline, urls, y, threshold, split_name, model_name):
    t0 = time.perf_counter()
    X = urls_to_feature_matrix(urls)
    extract_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    proba = pipeline.predict_proba(X)[:, 1]
    infer_time = time.perf_counter() - t1

    y_pred = (proba >= threshold).astype(int)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["legitimate", "phishing"])

    print(f"\n── {split_name} ──────────────────────────────────────")
    print(f"  N={len(y):,}  positive={y.sum():,} ({y.mean():.3f})")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Inference : {infer_time/len(y)*1000:.3f} ms/sample")
    print(f"\n{report}")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=["legitimate", "phishing"],
        yticklabels=["legitimate", "phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"URL Model – {split_name}\n{model_name}")
    fig.tight_layout()
    slug = split_name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(FIGURES / f"url_confusion_{slug}.png", dpi=150)
    plt.close(fig)

    return {
        "split": split_name,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference_ms_per_sample": infer_time / len(y) * 1000,
    }


def main() -> None:
    print("Loading URL model …")
    pipeline, metadata = load_model()
    threshold = metadata.get("threshold", 0.5)
    model_name = metadata.get("model_name", "unknown")
    print(f"Model: {model_name}  threshold: {threshold:.4f}")

    all_metrics = {"model": model_name, "threshold": threshold, "splits": {}}

    # Random test split
    test_path = PROCESSED / "urls_test.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        m = evaluate_split(
            pipeline, df["url"].tolist(), df["label"].values,
            threshold, "Random Test", model_name
        )
        all_metrics["splits"]["random_test"] = m
    else:
        print(f"[WARN] {test_path} not found.")

    # Domain-aware test split
    domain_test = PROCESSED / "urls_domain_test.csv"
    if domain_test.exists():
        df2 = pd.read_csv(domain_test)
        m2 = evaluate_split(
            pipeline, df2["url"].tolist(), df2["label"].values,
            threshold, "Domain-Aware Test", model_name
        )
        all_metrics["splits"]["domain_test"] = m2
        print(
            "\n[NOTE] Random-split and domain-aware results measure different "
            "things and must NOT be averaged or treated interchangeably.\n"
            "Domain-aware tests assess generalization to unseen domains."
        )
    else:
        print(f"[INFO] No domain-aware test split found at {domain_test}.")

    out_path = REPORTS / "url_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nURL metrics saved: {out_path}")


if __name__ == "__main__":
    main()
