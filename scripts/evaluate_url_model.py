"""
evaluate_url_model.py
=====================
Evaluates the URL phishing model honestly, on four separate test sets, using the
model that is actually allowed to see each one.

  Random Test          -> production model (trained on a random split)
  Domain-Aware Test    -> domain-holdout model (never saw these domains)
  Template-Clean Test  -> domain-holdout model (no shared path templates)
  LegitPhish External  -> both models (a different dataset entirely)

Why four: the production model is trained on a random split of PHIUSIL, and that
split shares most of the domain-aware test URLs (see
reports/url_leakage_report.json). Scoring the production model on the domain test
set would measure memorisation, not generalisation. The domain-holdout model is
trained only on the domain-isolated pool.

The LegitPhish set (Dataset C) is a different dataset from a different source
that is never used for training. It carries its own caveat: ~98.8% of its
*legitimate* URLs are byte-identical to PHIUSIL's, so precision on it is
optimistic; its *phishing* class has zero overlap with PHIUSIL and is a genuine
generalisation test. The evaluation therefore reports recall on unseen-domain
phishing separately, and the achievability of the two caveats is recorded in the
output rather than hidden.

Writes:
    reports/url_metrics.json
    reports/url_false_cases.csv
    reports/figures/url_confusion_*.png

Usage (from repo root):
    python scripts/evaluate_url_model.py
"""

import json
import pathlib
import re
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

from fraud_phishing.models.url_model import (
    load_domain_holdout_model,
    load_model,
    urls_to_feature_matrix,
)

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def evaluate_split(pipeline, urls, y, threshold, split_name, model_name):
    """
    Score one test set and return (metrics, false_case_rows).

    Positive class is phishing (label 1). False positives are legitimate URLs
    flagged as phishing; false negatives are phishing URLs that slipped through.
    """
    t0 = time.perf_counter()
    X = urls_to_feature_matrix(urls)
    extract_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    proba = pipeline.predict_proba(X)[:, 1]
    infer_time = time.perf_counter() - t1

    y_pred = (proba >= threshold).astype(int)
    f1 = f1_score(y, y_pred, zero_division=0)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    report = classification_report(
        y, y_pred, target_names=["legitimate", "phishing"], zero_division=0
    )

    print(f"\n── {split_name} ──────────────────────────────────────")
    print(f"  Evaluated model: {model_name}")
    print(f"  N={len(y):,}  positive={int(y.sum()):,} ({y.mean():.3f})")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Errors    : {fp:,} false positives (legit -> phishing), "
          f"{fn:,} false negatives (phishing -> legit)")
    print(f"  Inference : {infer_time/len(y)*1000:.3f} ms/sample")
    print(f"\n{report}")

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
    slug = re.sub(r"[^a-z0-9]+", "_", split_name.lower()).strip("_")
    fig.savefig(FIGURES / f"url_confusion_{slug}.png", dpi=150)
    plt.close(fig)

    metrics = {
        "split": split_name,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "confusion_matrix": {"labels": ["legitimate", "phishing"], "matrix": cm.tolist()},
        "classification_report": report,
        "inference_ms_per_sample": infer_time / len(y) * 1000,
        "feature_extraction_ms_per_sample": extract_time / len(y) * 1000,
    }

    false_cases = [
        {
            "split": split_name,
            "model": model_name,
            "url": u,
            "true_label": int(t),
            "true_label_name": "phishing" if int(t) == 1 else "legitimate",
            "predicted_label": int(p),
            "predicted_label_name": "phishing" if int(p) == 1 else "legitimate",
            "phishing_probability": round(float(pr), 6),
            "threshold": round(float(threshold), 6),
            "error_type": "false_positive" if int(p) == 1 else "false_negative",
        }
        for u, t, pr, p in zip(urls, y, proba, y_pred)
        if int(t) != int(p)
    ]

    return metrics, false_cases


def evaluate_external(external_path, candidate_models, all_false_cases):
    """
    Score every candidate model on the external LegitPhish set.

    LegitPhish was never used for training. Two things must be reported honestly:

      1. LegitPhish's *legitimate* class is ~98.8% byte-identical to PHIUSIL's
         legitimate URLs, so it is NOT an independent sample for that class and
         the precision numbers here are optimistic. This is recorded in the
         output rather than hidden.
      2. Its *phishing* class has zero exact-URL overlap with PHIUSIL and is
         ~98% unseen registered domains, so recall on it is a genuine
         generalisation test. That number is reported separately.
    """
    ext = pd.read_csv(external_path)
    ext["seen_in_phiusil"] = ext["seen_in_phiusil"].astype(str).str.lower() == "true"

    stats_path = REPORTS / "legitphish_external_stats.json"
    ext_stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else {}
    )

    unseen = ext[~ext["seen_in_phiusil"]].reset_index(drop=True)
    unseen_phish = unseen[unseen["label"] == 1].reset_index(drop=True)

    print(f"\n{'=' * 62}")
    print("EXTERNAL EVALUATION — LegitPhish (never used for training)")
    print(f"{'=' * 62}")
    print(
        f"  {len(ext):,} URLs | legitimate={int((ext['label'] == 0).sum()):,} "
        f"phishing={int((ext['label'] == 1).sum()):,}"
    )
    print(
        f"  unseen-domain subset: {len(unseen):,} | "
        f"unseen-domain phishing: {len(unseen_phish):,}"
    )
    print("  [CAVEAT] ~98.8% of LegitPhish's legitimate URLs are byte-identical to")
    print("           PHIUSIL URLs — the legitimate half is NOT independent.")

    external = {
        "dataset": "LegitPhish",
        "role": "external evaluation only (never used for training)",
        "source": "https://data.mendeley.com/datasets/hx4m73v2sf/2",
        "caveats": [
            "LegitPhish's legitimate class is ~98.8% byte-identical to PHIUSIL's "
            "legitimate URLs; precision on this set is therefore optimistic and "
            "must not be read as out-of-distribution performance.",
            "LegitPhish's phishing class has zero exact-URL overlap with PHIUSIL "
            "and ~98% unseen registered domains; recall on it is a genuine "
            "generalisation test.",
        ],
        "stats": ext_stats,
        "results": {},
    }

    for m_name, m_pipe, m_thr, m_src in candidate_models:
        res, cases = evaluate_split(
            m_pipe,
            ext["url"].tolist(),
            ext["label"].values,
            m_thr,
            f"LegitPhish External [{m_src}]",
            m_name,
        )
        all_false_cases += cases

        if len(unseen_phish):
            proba = m_pipe.predict_proba(
                urls_to_feature_matrix(unseen_phish["url"].tolist())
            )[:, 1]
            caught = int((proba >= m_thr).sum())
            recall_unseen = caught / len(unseen_phish)
            mean_p = float(proba.mean())
        else:
            caught, recall_unseen, mean_p = 0, float("nan"), float("nan")

        res["evaluated_model"] = m_name
        res["evaluated_model_source"] = m_src
        res["interpretation"] = (
            "external; legitimate class overlaps PHIUSIL so precision is "
            "optimistic, but phishing class is genuinely unseen"
        )
        res["unseen_domain_phishing_recall"] = recall_unseen
        res["unseen_domain_phishing_caught"] = caught
        res["unseen_domain_phishing_total"] = int(len(unseen_phish))
        res["unseen_domain_phishing_mean_probability"] = mean_p
        external["results"][m_src] = res

        print(f"\n  model={m_name} ({m_src})")
        print(
            f"    all URLs : P={res['precision']:.4f} R={res['recall']:.4f} "
            f"F1={res['f1']:.4f} AUC={res['roc_auc']:.4f} "
            f"FP={res['false_positives']:,} FN={res['false_negatives']:,}"
        )
        print(
            f"    unseen-domain phishing recall: {recall_unseen:.4f} "
            f"({caught:,}/{len(unseen_phish):,})  mean p={mean_p:.4f}"
        )

    return external


def main() -> None:
    print("Loading production URL model …")
    pipeline, metadata = load_model()
    threshold = float(metadata.get("threshold", 0.5))
    model_name = metadata.get("model_name", "unknown")
    print(f"Model: {model_name}  threshold: {threshold:.4f}")
    print("Trained on: urls_train + urls_val (random split)")

    all_metrics = {
        "label_map": {"0": "legitimate", "1": "phishing"},
        "positive_class": "phishing",
        "production_model": {
            "model": model_name,
            "threshold": threshold,
            "trained_on": "urls_train + urls_val (random split)",
        },
        "splits": {},
    }
    all_false_cases: list[dict] = []

    # ── 1. Random test: only the production model may see this ────────────
    test_path = PROCESSED / "urls_test.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        m, cases = evaluate_split(
            pipeline, df["url"].tolist(), df["label"].values,
            threshold, "Random Test", model_name,
        )
        m["evaluated_model"] = model_name
        m["evaluated_model_source"] = "production (random split)"
        m["interpretation"] = "in-distribution; domains and templates seen in training"
        all_metrics["splits"]["random_test"] = m
        all_false_cases += cases
    else:
        print(f"[WARN] {test_path} not found.")

    # ── 2 + 3. Unseen-domain tests: domain-holdout model only ─────────────
    try:
        domain_pipeline, domain_meta = load_domain_holdout_model()
    except FileNotFoundError as exc:
        print(f"\n[WARN] {exc}")
        domain_pipeline, domain_meta = None, {}

    d_name = domain_meta.get("model_name", "unknown") if domain_pipeline is not None else ""
    d_threshold = float(domain_meta.get("threshold", 0.5)) if domain_pipeline is not None else 0.5

    # Candidate (model, threshold, description) list used for external scoring.
    candidate_models: list[tuple[str, object, float, str]] = [
        (model_name, pipeline, threshold, "production (random split)"),
    ]

    if domain_pipeline is not None:
        all_metrics["domain_holdout_model"] = {
            "model": d_name,
            "threshold": d_threshold,
            "trained_on": "urls_template_train.csv (domain-isolated pool)",
            "threshold_selection": domain_meta.get("threshold_selection", {}),
        }
        candidate_models.append(
            (d_name, domain_pipeline, d_threshold, "domain-holdout")
        )
        print(
            f"\nDomain-holdout model: {d_name}  threshold: {d_threshold:.4f} "
            "(trained without any domain-aware test data)"
        )

        for split_name, filename, key, note in (
            (
                "Domain-Aware Test",
                "urls_domain_test.csv",
                "domain_test",
                "registered domains disjoint from the training pool",
            ),
            (
                "Template-Clean Test",
                "urls_template_test.csv",
                "template_clean_test",
                "domains disjoint AND no shared URL path template with training",
            ),
        ):
            path = PROCESSED / filename
            if not path.exists():
                print(f"[INFO] No {filename} found; skipping {split_name}.")
                continue
            df2 = pd.read_csv(path)
            m2, cases = evaluate_split(
                domain_pipeline, df2["url"].tolist(), df2["label"].values,
                d_threshold, split_name, d_name,
            )
            m2["evaluated_model"] = d_name
            m2["evaluated_model_source"] = "domain-holdout (domain-isolated pool)"
            m2["interpretation"] = f"unseen-domain; {note}"
            all_metrics["splits"][key] = m2
            all_false_cases += cases

    # ── 4. External test: LegitPhish (never used for training) ────────────
    external_path = PROCESSED / "legitphish_external.csv"
    if external_path.exists():
        external = evaluate_external(
            external_path, candidate_models, all_false_cases
        )
        all_metrics["external_legitphish"] = external
    else:
        print(
            f"\n[INFO] No external LegitPhish set at {external_path}; skipping.\n"
            "       Run: python src/fraud_phishing/data/prepare_external_urls.py"
        )

    # ── Leakage context ───────────────────────────────────────────────────
    leakage_path = REPORTS / "url_leakage_report.json"
    if leakage_path.exists():
        all_metrics["leakage_report"] = json.loads(leakage_path.read_text(encoding="utf-8"))
        print(f"\nLeakage report embedded from: {leakage_path}")

    print(
        "\n[NOTE] The three splits measure different things and must not be "
        "averaged or treated interchangeably:\n"
        "  * Random Test is in-distribution (domains/templates are shared with "
        "training) -- it estimates serving-time accuracy on familiar patterns.\n"
        "  * Domain-Aware and Template-Clean Tests measure generalisation to "
        "unseen domains/templates, which is the harder and more honest question."
    )

    out_path = REPORTS / "url_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nURL metrics saved: {out_path}")

    cases_path = REPORTS / "url_false_cases.csv"
    if all_false_cases:
        pd.DataFrame(all_false_cases).to_csv(cases_path, index=False, encoding="utf-8")
        print(f"False cases saved: {cases_path} ({len(all_false_cases):,} rows)")
    else:
        print("No misclassified URLs.")


if __name__ == "__main__":
    main()
