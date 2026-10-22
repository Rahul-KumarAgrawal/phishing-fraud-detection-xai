"""
prepare_urls_augmented.py
=========================
Builds the augmented URL training pool used by the revised model.

Why augmentation is needed
--------------------------
PHIUSIL's legitimate class is almost entirely bare ``https://domain`` URLs with
no path (see ``reports/url_diagnosis.md``). A model trained on it alone learns
"URL has a path -> phishing", which is not a phishing signal at all. The fix is
to ensure the *training* data contains realistic legitimate URLs with paths,
query strings, subdomains and mixed schemes, alongside diverse phishing.

Composition
-----------
    base    : data/processed/urls_template_train.csv
              (PHIUSIL, domain-isolated pool already disjoint from the
               PHIUSIL template-clean / domain-aware test sets)
    augment : data/raw/MURL_urls.csv
              (Dataset D -- a different corpus with 0 byte-identical URLs to
               PHIUSIL, 86% path-bearing; benign / legitimate -> 0,
               phishing -> 1; defacement / malware / spam / exploit dropped)

Datasets C (LegitPhish) and E (URL-Phish) are deliberately NOT in this pool.
LegitPhish is the development benchmark and URL-Phish is the untouched final
external set; neither may be trained on.

Splitting
---------
The combined pool is split by **registered domain** into a fitting pool (85%) and
a threshold-selection slice (15%), so the threshold is always chosen on unseen
domains. URLs whose registered domain appears in a PHIUSIL evaluation split are
dropped from the augmentation so those splits stay valid.

Writes:
    data/processed/urls_aug_train.csv
    data/processed/urls_aug_val.csv
    data/processed/urls_aug_split_stats.json

Usage (from repo root):
    python src/fraud_phishing/data/prepare_urls_augmented.py
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

PROCESSED = pathlib.Path("data/processed")
RAW = pathlib.Path("data/raw")
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

PHIUSIL_POOL = PROCESSED / "urls_template_train.csv"
MURL_RAW = RAW / "MURL_urls.csv"

# PHIUSIL evaluation splits that must stay domain-disjoint from all training data.
# NOTE: urls_test.csv is deliberately excluded -- it is the *random* split and
# shares registered domains with the training pool by construction (that is the
# in-distribution test). Only the domain-aware splits are required to be unseen.
PHIUSIL_EVAL_SPLITS = (
    "urls_template_test.csv",
    "urls_domain_test.csv",
)

# Caps keep training time bounded while retaining diversity.
MURL_BENIGN_CAP = 60_000
MURL_PHISHING_CAP = 60_000

OUT = PROCESSED / "urls_aug_train.csv"
OUT_VAL = PROCESSED / "urls_aug_val.csv"
OUT_STATS = PROCESSED / "urls_aug_split_stats.json"


def _reg(url: str) -> str:
    return tldextract.extract(str(url)).top_domain_under_public_suffix or "__none__"


def main() -> None:
    if not PHIUSIL_POOL.exists():
        print(f"[ERROR] {PHIUSIL_POOL} not found. Run prepare_urls.py first.")
        sys.exit(1)
    if not MURL_RAW.exists():
        print(f"[ERROR] {MURL_RAW} not found. Run download_data.py first.")
        sys.exit(1)

    print("=" * 62)
    print("Building augmented URL training pool (PHIUSIL + MURL)")
    print("=" * 62)

    base = pd.read_csv(PHIUSIL_POOL)
    base = base[["url", "label"]].copy()
    base["source"] = "phiusil"
    print(f"PHIUSIL base pool: {len(base):,} "
          f"(legit={int((base['label'] == 0).sum()):,}, "
          f"phish={int((base['label'] == 1).sum()):,})")

    murl = pd.read_csv(MURL_RAW)
    murl = murl[["url", "label"]].copy()
    murl["url"] = murl["url"].astype(str).str.strip()
    murl = murl[murl["url"] != ""]

    # ── Drop anything already present in PHIUSIL (exact URL) ──────────────
    phiusil_all = set()
    for part in PROCESSED.glob("urls_*.csv"):
        if part.name in ("urls_aug_train.csv", "urls_aug_val.csv"):
            continue
        phiusil_all |= set(pd.read_csv(part)["url"].astype(str))
    before = len(murl)
    murl = murl[~murl["url"].isin(phiusil_all)]
    print(f"MURL rows dropped as exact PHIUSIL duplicates: {before - len(murl):,}")

    murl = murl.drop_duplicates(subset=["url"])

    # ── Drop MURL rows whose domain collides with a PHIUSIL eval split ────
    eval_domains: set[str] = set()
    for name in PHIUSIL_EVAL_SPLITS:
        p = PROCESSED / name
        if p.exists():
            eval_domains |= {_reg(u) for u in pd.read_csv(p)["url"].astype(str)}
    if eval_domains:
        murl_dom = murl["url"].map(_reg)
        before = len(murl)
        murl = murl[~murl_dom.isin(eval_domains)]
        print(
            f"MURL rows dropped to keep PHIUSIL eval splits domain-disjoint: "
            f"{before - len(murl):,}"
        )

    # ── Cap each class ────────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    parts = []
    for label, cap in ((0, MURL_BENIGN_CAP), (1, MURL_PHISHING_CAP)):
        sub = murl[murl["label"] == label]
        if len(sub) > cap:
            idx = rng.choice(len(sub), cap, replace=False)
            sub = sub.iloc[np.sort(idx)]
        print(f"MURL class {label}: {len(sub):,} kept (cap {cap:,})")
        parts.append(sub)
    murl = pd.concat(parts, ignore_index=True)
    murl["source"] = "murl"

    combined = pd.concat([base, murl], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    combined["domain"] = combined["url"].map(_reg)
    print(
        f"\nCombined pool: {len(combined):,} "
        f"(legit={int((combined['label'] == 0).sum()):,}, "
        f"phish={int((combined['label'] == 1).sum()):,})"
    )

    # ── Domain-isolated fit / validation split ────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    fit_idx, val_idx = next(
        gss.split(combined, combined["label"], groups=combined["domain"])
    )
    fit, val = combined.iloc[fit_idx], combined.iloc[val_idx]
    overlap = (set(fit["domain"]) - {"__none__"}) & (set(val["domain"]) - {"__none__"})
    if overlap:
        raise RuntimeError(f"fit/val share {len(overlap)} registered domains")
    print("[OK] No registered domain appears in both fit and validation slices.")

    # ── Assert the PHIUSIL eval splits are still domain-disjoint ──────────
    train_domains = set(combined["domain"]) - {"__none__"}
    leak = train_domains & eval_domains
    if leak:
        raise RuntimeError(
            f"Training pool shares {len(leak)} registered domains with a PHIUSIL "
            "evaluation split; those splits would no longer be unseen-domain."
        )
    print("[OK] Training pool is domain-disjoint from every PHIUSIL eval split.")

    fit[["url", "label"]].to_csv(OUT, index=False)
    val[["url", "label"]].to_csv(OUT_VAL, index=False)

    def _shapes(df):
        return {
            "total": int(len(df)),
            "legitimate": int((df["label"] == 0).sum()),
            "phishing": int((df["label"] == 1).sum()),
            "from_phiusil": int((df["source"] == "phiusil").sum()),
            "from_murl": int((df["source"] == "murl").sum()),
        }

    stats = {
        "random_seed": RANDOM_SEED,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "purpose": (
            "Augmented training pool: PHIUSIL domain-isolated pool + MURL "
            "benign/legitimate/phishing. MURL supplies realistic path-bearing "
            "legitimate URLs that PHIUSIL lacks."
        ),
        "sources": {
            "phiusil": "data/processed/urls_template_train.csv (Dataset B pool)",
            "murl": "data/raw/MURL_urls.csv (Dataset D, augmentation only)",
        },
        "not_used_for_training": [
            "LegitPhish (Dataset C) - development benchmark",
            "URL-Phish (Dataset E) - untouched final external set",
        ],
        "caps": {"murl_benign": MURL_BENIGN_CAP, "murl_phishing": MURL_PHISHING_CAP},
        "split": "GroupShuffleSplit by registered domain (85/15), seed 42",
        "fit": _shapes(fit),
        "val": _shapes(val),
        "phiusil_eval_splits_kept_disjoint": list(PHIUSIL_EVAL_SPLITS),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nWrote {OUT} ({len(fit):,})")
    print(f"Wrote {OUT_VAL} ({len(val):,})")
    print(f"Wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
