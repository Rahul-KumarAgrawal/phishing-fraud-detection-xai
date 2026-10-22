"""
compare_url_models.py
=====================
Step 4 of the P3 URL-robustness work: compare the original and revised URL
models side by side on every evaluation set, with the *same* metrics.

Models
------
    baseline (production)   all 33 features, trained on the PHIUSIL random split
    baseline (domain-heldout) all 33 features, trained on the PHIUSIL domain-isolated pool
    revised                 selected feature subset, trained on the augmented pool

Evaluation sets
---------------
    PHIUSIL random test         in-distribution (shares domains with training)
    PHIUSIL template-clean test unseen domains AND no shared path template
    LegitPhish (dev benchmark)  used throughout development
    URL-Phish (FINAL external)  touched exactly once, here, for the final claim

Reported for every pair: precision, recall, F1, ROC-AUC, confusion matrix, FP and
FN counts, false-positive rate on legitimate URLs and false-negative rate on
phishing URLs.

Writes:
    reports/url_model_comparison.md
    reports/url_model_comparison.json

Usage (from repo root):
    python scripts/compare_url_models.py
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.url_model import (
    load_domain_holdout_model,
    load_model,
    load_revised_model,
    urls_to_feature_matrix,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)


def metrics(y, proba, threshold) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())
    n_legit, n_phish = tn + fp, fn + tp
    try:
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "n_legitimate": n_legit,
        "n_phishing": n_phish,
        "threshold": round(float(threshold), 4),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                             "layout": "[[tn, fp], [fn, tp]]"},
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / max(n_legit, 1),
        "false_negative_rate": fn / max(n_phish, 1),
    }


def main() -> None:
    models = {}
    try:
        p, m = load_model()
        models["baseline (production)"] = (p, m)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
    try:
        p, m = load_domain_holdout_model()
        models["baseline (domain-heldout)"] = (p, m)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
    try:
        p, m = load_revised_model()
        models["revised"] = (p, m)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")

    if not models:
        print("No models available; run scripts/train_url_model.py first.")
        return

    for name, (_, m) in models.items():
        print(f"  {name:28s} model={m.get('model_name')} "
              f"n_features={m.get('n_features')} "
              f"subset={m.get('feature_subset', 'all')} thr={float(m.get('threshold', 0.5)):.4f}")

    # ── Evaluation sets ───────────────────────────────────────────────────
    sets: dict[str, tuple[list[str], np.ndarray, dict]] = {}
    for key, fname in (
        ("phiusil_random_test", "urls_test.csv"),
        ("phiusil_template_test", "urls_template_test.csv"),
    ):
        p = PROCESSED / fname
        if p.exists():
            df = pd.read_csv(p)
            sets[key] = (df["url"].astype(str).tolist(),
                         df["label"].astype(int).values, {})

    dev_path = PROCESSED / "legitphish_external.csv"
    if dev_path.exists():
        df = pd.read_csv(dev_path)
        sets["legitphish_dev"] = (df["url"].astype(str).tolist(),
                                  df["label"].astype(int).values, {})

    ext_path = PROCESSED / "urlphish_external.csv"
    if ext_path.exists():
        df = pd.read_csv(ext_path)
        extra: dict = {}
        if "byte_identical_to_training" in df.columns:
            extra["byte_identical"] = (
                df["byte_identical_to_training"].astype(str).str.lower() == "true"
            ).values
        if "seen_in_phiusil_domain" in df.columns:
            extra["unseen_domain"] = (
                df["seen_in_phiusil_domain"].astype(str).str.lower() != "true"
            ).values
        sets["urlphish_final_external"] = (df["url"].astype(str).tolist(),
                                           df["label"].astype(int).values, extra)

    # PHIUSIL pool URLs, used to split the dev benchmark by byte-identity.
    pool = set()
    for part in PROCESSED.glob("urls_*.csv"):
        if part.name.startswith("urls_aug"):
            continue
        pool |= set(pd.read_csv(part)["url"].astype(str))

    results: dict[str, dict] = {}
    for set_name, (urls, y, extra) in sets.items():
        results[set_name] = {}
        for model_name, (pipe, meta) in models.items():
            feats = meta.get("feature_names")
            thr = float(meta.get("threshold", 0.5))
            X = urls_to_feature_matrix(urls, feature_names=feats)
            proba = pipe.predict_proba(X)[:, 1]
            r = metrics(y, proba, thr)
            r["model_name"] = meta.get("model_name")
            r["n_features"] = meta.get("n_features")
            r["feature_subset"] = meta.get("feature_subset", "all")

            # Stratified views that matter for this project.
            is_url = np.array([u in pool for u in urls])
            if set_name == "legitphish_dev":
                new_legit = (y == 0) & ~is_url
                unseen_phish = (y == 1) & ~is_url
                r["new_legitimate_n"] = int(new_legit.sum())
                r["new_legitimate_fpr"] = float((proba[new_legit] >= thr).mean()) \
                    if new_legit.any() else float("nan")
                r["unseen_phishing_n"] = int(unseen_phish.sum())
                r["unseen_phishing_recall"] = float((proba[unseen_phish] >= thr).mean()) \
                    if unseen_phish.any() else float("nan")
            if set_name == "urlphish_final_external":
                schemes = np.array([
                    "https" if u.lower().startswith("https://")
                    else ("http" if u.lower().startswith("http://") else "none")
                    for u in urls
                ])
                for cls, cname in ((0, "benign"), (1, "phishing")):
                    m = y == cls
                    r[f"{cname}_n"] = int(m.sum())
                    r[f"{cname}_rate"] = float((proba[m] >= thr).mean())
                for sc in ("https", "http"):
                    m = (y == 1) & (schemes == sc)
                    if m.any():
                        r[f"phishing_{sc}_n"] = int(m.sum())
                        r[f"phishing_{sc}_recall"] = float((proba[m] >= thr).mean())
                m = (y == 0) & (schemes == "https")
                if m.any():
                    r["benign_https_n"] = int(m.sum())
                    r["benign_https_fpr"] = float((proba[m] >= thr).mean())

                # Honest strata: rows that are byte-identical to training data
                # cannot test generalisation, and rows on a domain the model was
                # trained on are weaker evidence than unseen-domain rows.
                if extra.get("byte_identical") is not None:
                    keep = ~extra["byte_identical"]
                    r["n_excluding_memorised"] = int(keep.sum())
                    r["metrics_excluding_memorised"] = metrics(y[keep], proba[keep], thr)
                if extra.get("unseen_domain") is not None:
                    keep = extra["unseen_domain"]
                    r["n_unseen_domain_only"] = int(keep.sum())
                    r["metrics_unseen_domain_only"] = metrics(y[keep], proba[keep], thr)
            results[set_name][model_name] = r

    # ── Write JSON ────────────────────────────────────────────────────────
    payload = {
        "note": (
            "LegitPhish (Dataset C) is the development benchmark: it and the "
            "diagnosis derived from it shaped the revised model. URL-Phish "
            "(Dataset E) was downloaded but held out of every decision and is "
            "evaluated here for the first time."
        ),
        "models": {n: {k: m.get(k) for k in
                       ("model_name", "n_features", "feature_subset",
                        "feature_schema_version", "threshold", "trained_on")}
                   for n, (_, m) in models.items()},
        "results": results,
    }
    (REPORTS / "url_model_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ── Write Markdown ────────────────────────────────────────────────────
    L: list[str] = []
    L.append("# URL Model Comparison — original vs revised\n")
    L.append("Reproduce with `python scripts/compare_url_models.py`.\n")
    L.append("Positive class is **phishing**. Confusion matrix layout is "
             "`[[tn, fp], [fn, tp]]`. FPR is on legitimate URLs, FNR on phishing URLs.\n")
    L.append("| model | subset | n features | threshold | trained on |")
    L.append("|---|---|---:|---:|---|")
    for n, (_, m) in models.items():
        L.append(f"| {n} | `{m.get('feature_subset', 'all')}` | {m.get('n_features')} | "
                 f"{float(m.get('threshold', 0.5)):.4f} | {m.get('trained_on', 'PHIUSIL')} |")
    L.append("")

    titles = {
        "phiusil_random_test": "PHIUSIL random test (in-distribution)",
        "phiusil_template_test": "PHIUSIL template-clean test (unseen domains + templates)",
        "legitphish_dev": "LegitPhish — development benchmark",
        "urlphish_final_external": "URL-Phish — FINAL external (untouched during development)",
    }
    for set_name, per_model in results.items():
        L.append(f"## {titles.get(set_name, set_name)}\n")
        L.append("| model | P | R | F1 | ROC-AUC | FP | FN | FPR | FNR | confusion |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for mn, r in per_model.items():
            cm = r["confusion_matrix"]
            L.append(
                f"| {mn} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                f"{r['roc_auc']:.4f} | {r['false_positives']:,} | {r['false_negatives']:,} | "
                f"{r['false_positive_rate']*100:.2f}% | {r['false_negative_rate']*100:.2f}% | "
                f"[[{cm['tn']}, {cm['fp']}], [{cm['fn']}, {cm['tp']}]] |")
        L.append("")
        if set_name == "legitphish_dev":
            L.append("Byte-identity stratification (the failure case from "
                     "`reports/url_diagnosis.md`):\n")
            L.append("| model | FPR on new legitimate URLs | recall on unseen phishing |")
            L.append("|---|---:|---:|")
            for mn, r in per_model.items():
                L.append(f"| {mn} | **{r['new_legitimate_fpr']*100:.1f}%** "
                         f"({r['new_legitimate_n']} URLs) | "
                         f"{r['unseen_phishing_recall']*100:.1f}% "
                         f"({r['unseen_phishing_n']} URLs) |")
            L.append("")
        if set_name == "urlphish_final_external":
            L.append("Stratified (this dataset's HTTPS rate is inverted relative to "
                     "PHIUSIL, so an `is_https` shortcut cannot pass it):\n")
            L.append("| model | benign FPR | phishing recall | phishing http recall | "
                     "phishing https recall |")
            L.append("|---|---:|---:|---:|---:|")
            for mn, r in per_model.items():
                L.append(
                    f"| {mn} | {r['benign_rate']*100:.2f}% | {r['phishing_rate']*100:.2f}% | "
                    f"{r.get('phishing_http_recall', float('nan'))*100:.2f}% | "
                    f"{r.get('phishing_https_recall', float('nan'))*100:.2f}% |")
            L.append("")
            L.append("Integrity strata — 595 benign URLs are byte-identical to PHIUSIL "
                     "training data (they cannot test generalisation), and 10,112 "
                     "phishing URLs sit on a domain seen in PHIUSIL:\n")
            L.append("| model | P (excl. memorised) | R (excl. memorised) | F1 (excl. memorised) "
                     "| P (unseen domain) | R (unseen domain) | F1 (unseen domain) |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for mn, r in per_model.items():
                a = r.get("metrics_excluding_memorised", {})
                b = r.get("metrics_unseen_domain_only", {})
                L.append(
                    f"| {mn} | {a.get('precision', float('nan')):.4f} | "
                    f"{a.get('recall', float('nan')):.4f} | {a.get('f1', float('nan')):.4f} | "
                    f"{b.get('precision', float('nan')):.4f} | "
                    f"{b.get('recall', float('nan')):.4f} | {b.get('f1', float('nan')):.4f} |")
            L.append("")

    L.append("## Which datasets influenced development\n")
    L.append("- **PHIUSIL (Dataset B)** — training data for the original model; also "
             "the source of the formatting shortcut identified in Step 1.")
    L.append("- **MURL (Dataset D)** — training augmentation only.")
    L.append("- **LegitPhish (Dataset C)** — development benchmark. The diagnosis and "
             "the choice of the revised feature subset came from it.")
    L.append("- **URL-Phish (Dataset E)** — untouched final external set. It was "
             "downloaded, its composition asserted, and then not inspected until the "
             "single evaluation above. No model, feature or threshold decision used it.")
    L.append("")
    L.append("No model was trained on LegitPhish or URL-Phish.")
    L.append("")

    (REPORTS / "url_model_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {REPORTS / 'url_model_comparison.md'}")
    print(f"Wrote {REPORTS / 'url_model_comparison.json'}")


if __name__ == "__main__":
    main()
