"""
ablate_url_features.py
======================
Step 2 of the P3 URL-robustness work: controlled feature ablation.

It trains the same pipelines on the same data with different *feature subsets*
and measures what actually changes, so feature decisions are made on evidence
rather than intuition. Nothing is removed blindly.

Two training pools are compared:
    phiusil    - data/processed/urls_template_train.csv  (the original regime)
    augmented  - data/processed/urls_aug_train.csv       (PHIUSIL + MURL)

Each pool's threshold is selected on a *domain-held-out* slice.

Evaluation sets (URL-Phish is deliberately NOT touched here -- it is the
untouched final external set):
    - PHIUSIL template-clean test (unseen domains)
    - LegitPhish development benchmark, split by byte-identity to PHIUSIL:
        * FPR on the 445 genuinely new legitimate URLs (the failure case)
        * recall on the 63,492 byte-distinct phishing URLs (62,482 of which are
          also on a domain PHIUSIL never contained)
        * F1 / AUC over the whole benchmark

Writes:
    reports/url_feature_ablation.json
    reports/url_feature_ablation.md
    .cache/url_features/*.npz   (feature-matrix cache; gitignored)

Usage (from repo root):
    python scripts/ablate_url_features.py
"""

import json
import hashlib
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.features.url_features import FEATURE_NAMES, extract_features
from fraud_phishing.features.feature_subsets import BRAND_FEATURES, SUBSETS
from fraud_phishing.models.url_model import (
    CANDIDATE_PIPELINES,
    select_threshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = pathlib.Path("data/processed")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)
CACHE = pathlib.Path(".cache/url_features")
CACHE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ABLATION_TRAIN_SAMPLES = 150_000
ABLATION_TREES = 150  # reduced vs production (200) purely for runtime


# ── Feature extraction with an on-disk cache ─────────────────────────────────

def features_for(csv_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, frame) for a processed split, cached on disk.

    The cache key includes a SHA-256 of the CSV, so editing a split (for example
    changing the augmentation) can never silently reuse stale features.
    """
    src = PROCESSED / csv_name
    cache = CACHE / (csv_name.replace(".csv", ".npz"))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    df = pd.read_csv(src)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        cached_digest = str(d["digest"]) if "digest" in d.files else None
        if (
            int(d["n"]) == len(df)
            and cached_digest == digest
            and list(d["names"]) == list(FEATURE_NAMES)
        ):
            return d["X"], df["label"].astype(int).values, df
    print(f"  Extracting features for {csv_name} ({len(df):,}) ...")
    t0 = time.perf_counter()
    X = np.array(
        [[extract_features(u)[n] for n in FEATURE_NAMES] for u in df["url"].astype(str)],
        dtype=float,
    )
    print(f"    done in {time.perf_counter()-t0:.1f}s -> {X.shape}")
    np.savez_compressed(cache, X=X, n=len(df), digest=digest,
                        names=np.array(FEATURE_NAMES))
    return X, df["label"].astype(int).values, df


def build_pipeline(kind: str) -> Pipeline:
    if kind == "RandomForest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=ABLATION_TREES, min_samples_split=5,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED, solver="lbfgs")),
    ])


def stratified_sample(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(y == cls)
        take = int(round(n * len(cls_idx) / len(y)))
        idx.append(rng.choice(cls_idx, min(take, len(cls_idx)), replace=False))
    return np.sort(np.concatenate(idx))


def binary_metrics(y, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        "false_positives": int(((y == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y == 1) & (y_pred == 0)).sum()),
        "false_positive_rate": float(((y == 0) & (y_pred == 1)).sum() / max((y == 0).sum(), 1)),
    }


def main() -> None:
    print("=" * 70)
    print("URL feature ablation")
    print("=" * 70)

    # ── Load feature matrices ─────────────────────────────────────────────
    print("\nLoading data ...")
    X_aug_fit, y_aug_fit, _ = features_for("urls_aug_train.csv")
    X_aug_val, y_aug_val, _ = features_for("urls_aug_val.csv")
    X_ph_train, y_ph_train, _ = features_for("urls_template_train.csv")
    X_ph_test, y_ph_test, _ = features_for("urls_template_test.csv")
    X_dev, y_dev, dev_df = features_for("legitphish_external.csv")

    # PHIUSIL pool: mirror the augmented protocol with a domain-held-out slice.
    import tldextract  # noqa: PLC0415
    from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

    ph_urls = pd.read_csv(PROCESSED / "urls_template_train.csv")["url"].astype(str)
    groups = np.array([tldextract.extract(u).top_domain_under_public_suffix or "__none__"
                       for u in ph_urls])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    ph_fit_idx, ph_val_idx = next(gss.split(X_ph_train, y_ph_train, groups=groups))

    pools = {
        "phiusil": (X_ph_train[ph_fit_idx], y_ph_train[ph_fit_idx],
                    X_ph_train[ph_val_idx], y_ph_train[ph_val_idx]),
        "augmented": (X_aug_fit, y_aug_fit, X_aug_val, y_aug_val),
    }

    # ── Dev-benchmark slice definitions ───────────────────────────────────
    pool_urls = set()
    for part in PROCESSED.glob("urls_*.csv"):
        if part.name.startswith("urls_aug"):
            continue
        pool_urls |= set(pd.read_csv(part)["url"].astype(str))
    is_new_legit = (y_dev == 0) & ~dev_df["url"].astype(str).isin(pool_urls).values
    unseen_phish = (y_dev == 1) & ~dev_df["url"].astype(str).isin(pool_urls).values
    print(f"\nDev benchmark: {len(y_dev):,} total | "
          f"{int(is_new_legit.sum())} new legitimate | "
          f"{int(unseen_phish.sum())} unseen phishing")

    feat_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    results = []

    for pool_name, (Xf, yf, Xv, yv) in pools.items():
        samp = stratified_sample(yf, ABLATION_TRAIN_SAMPLES)
        Xf, yf = Xf[samp], yf[samp]
        for subset_name, cols in SUBSETS.items():
            col_idx = [feat_index[c] for c in cols]
            for kind in ("RandomForest", "LogisticRegression"):
                tag = f"{pool_name}/{subset_name}/{kind}"
                t0 = time.perf_counter()
                pipe = build_pipeline(kind)
                pipe.fit(Xf[:, col_idx], yf)
                thr = select_threshold(pipe, Xv[:, col_idx], yv)

                r = {
                    "pool": pool_name,
                    "subset": subset_name,
                    "n_features": len(cols),
                    "model": kind,
                    "fit_seconds": round(time.perf_counter() - t0, 1),
                }
                r["val"] = binary_metrics(yv, pipe.predict_proba(Xv[:, col_idx])[:, 1], thr)

                p_ph = pipe.predict_proba(X_ph_test[:, col_idx])[:, 1]
                r["phiusil_template_test"] = binary_metrics(y_ph_test, p_ph, thr)

                p_dev = pipe.predict_proba(X_dev[:, col_idx])[:, 1]
                r["dev"] = binary_metrics(y_dev, p_dev, thr)
                r["dev"]["new_legitimate_fpr"] = float(
                    (p_dev[is_new_legit] >= thr).mean())
                r["dev"]["new_legitimate_flagged"] = int((p_dev[is_new_legit] >= thr).sum())
                r["dev"]["new_legitimate_n"] = int(is_new_legit.sum())
                r["dev"]["unseen_phishing_recall"] = float(
                    (p_dev[unseen_phish] >= thr).mean())
                r["dev"]["unseen_phishing_n"] = int(unseen_phish.sum())

                results.append(r)
                print(
                    f"  {tag:46s} nfeat={len(cols):>2} "
                    f"valF1={r['val']['f1']:.4f} "
                    f"phTestF1={r['phiusil_template_test']['f1']:.4f} "
                    f"newLegitFPR={r['dev']['new_legitimate_fpr']*100:6.1f}% "
                    f"unseenPhishR={r['dev']['unseen_phishing_recall']*100:5.1f}% "
                    f"({r['fit_seconds']}s)"
                )

    (REPORTS / "url_feature_ablation.json").write_text(
        json.dumps({"feature_names": list(FEATURE_NAMES), "results": results}, indent=2),
        encoding="utf-8",
    )

    # ── Markdown report ───────────────────────────────────────────────────
    lines = [
        "# URL Feature Ablation\n",
        "Reproduce with `python scripts/ablate_url_features.py`.\n",
        f"Training pools: **phiusil** (`urls_template_train.csv`) and "
        f"**augmented** (`urls_aug_train.csv`, PHIUSIL + MURL). "
        f"Both use a domain-held-out threshold slice. Models are fitted on a "
        f"stratified {ABLATION_TRAIN_SAMPLES:,} sample; RandomForest uses "
        f"{ABLATION_TREES} trees here (production uses 200) for runtime.\n",
        "**URL-Phish is not used here** - it is the untouched final external set.\n",
        "`newLegitFPR` = false-positive rate on the 445 legitimate URLs that are "
        "byte-distinct from PHIUSIL (the failure case in `reports/url_diagnosis.md`). "
        "`unseenPhishR` = recall on the 63,492 phishing URLs that are byte-distinct "
        "from PHIUSIL (62,482 of them are also on a domain PHIUSIL never had).\n",
        "| pool | subset | n feat | model | val F1 | PHIUSIL test F1 | dev F1 | "
        "**new legit FPR** | unseen phish recall |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['pool']} | {r['subset']} | {r['n_features']} | {r['model']} | "
            f"{r['val']['f1']:.4f} | {r['phiusil_template_test']['f1']:.4f} | "
            f"{r['dev']['f1']:.4f} | **{r['dev']['new_legitimate_fpr']*100:.1f}%** | "
            f"{r['dev']['unseen_phishing_recall']*100:.1f}% |"
        )
    lines.append("")
    (REPORTS / "url_feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {REPORTS / 'url_feature_ablation.json'}")
    print(f"Wrote {REPORTS / 'url_feature_ablation.md'}")


if __name__ == "__main__":
    main()
