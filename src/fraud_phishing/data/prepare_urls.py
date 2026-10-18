"""
prepare_urls.py
===============
Prepares the PHIUSIL raw URL dataset for model training.

Steps:
  1. Load raw CSV (produced by download_data.py).
  2. Verify and normalize labels: legitimate → 0, phishing → 1.
  3. Drop missing/empty URLs and exact duplicates.
  4. Stratified train / val / test split (70 / 15 / 15).
  5. Also build a domain-aware test split for leakage evaluation.
  6. Save splits to data/processed/.

Usage (from repo root):
    python src/fraud_phishing/data/prepare_urls.py
"""

import json
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# tldextract imported lazily (used only for domain grouping)

RAW_URL = pathlib.Path("data/raw/PHIUSIL_urls.csv")
PROCESSED = pathlib.Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── Label normalization ────────────────────────────────────────────────────
# Expected raw values from PHIUSIL: 1 = phishing, 0 = legitimate
# (verify against dataset documentation)
# If raw labels are -1/1, they are re-mapped below.
PHIUSIL_LABEL_MAP_STR = {
    "0": 0,   # legitimate
    "1": 1,   # phishing
    "-1": 0,  # some datasets use -1 for legitimate
}
PHIUSIL_LABEL_MAP_INT = {
    0: 0,
    1: 1,
    -1: 0,
}


def load_raw(path: pathlib.Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"Loaded {path.name} ({enc}): {len(df):,} rows, cols={list(df.columns)}")
            return df
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path}")


def normalize_label(series: pd.Series) -> pd.Series:
    """Map raw labels to {0, 1}. Raises if unknown values are encountered."""
    str_map = {str(k): v for k, v in PHIUSIL_LABEL_MAP_STR.items()}
    int_map = PHIUSIL_LABEL_MAP_INT

    def _map(v):
        sv = str(v).strip()
        if sv in str_map:
            return str_map[sv]
        raise ValueError(f"Unknown label value: {v!r}")

    return series.apply(_map).astype(int)


def prepare(path: pathlib.Path = RAW_URL) -> None:
    if not path.exists():
        print(f"[ERROR] Raw URL dataset not found at: {path}")
        print("Run: python src/fraud_phishing/data/download_data.py")
        sys.exit(1)

    df = load_raw(path)

    # ── Find URL and label columns ─────────────────────────────────────────
    col_lower = {c.lower().strip(): c for c in df.columns}
    url_col = None
    for cand in ("url", "urls", "link"):
        if cand in col_lower:
            url_col = col_lower[cand]
            break

    label_col = None
    for cand in ("label_raw", "label", "target", "class", "phishing"):
        if cand in col_lower:
            label_col = col_lower[cand]
            break

    if url_col is None or label_col is None:
        raise ValueError(
            f"Cannot identify URL/label columns from: {list(df.columns)}\n"
            "Update column detection in prepare_urls.py."
        )
    print(f"URL column: '{url_col}', Label column: '{label_col}'")

    raw_counts = df[label_col].value_counts(dropna=False)
    print(f"\nRaw label distribution:\n{raw_counts.to_string()}")

    # ── Normalize labels ───────────────────────────────────────────────────
    try:
        df["label"] = normalize_label(df[label_col])
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("Check label mapping in prepare_urls.py:PHIUSIL_LABEL_MAP_STR")
        sys.exit(1)

    # ── Drop missing / empty URLs ──────────────────────────────────────────
    pre = len(df)
    df["url"] = df[url_col].astype(str).str.strip()
    df = df[df["url"] != ""]
    df = df[df["url"].str.lower() != "nan"]
    print(f"\nDropped empty/NaN URLs: {pre - len(df):,}")

    # ── Drop exact duplicate URL strings ──────────────────────────────────
    pre_dup = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"Dropped exact duplicate URLs: {pre_dup - len(df):,}")

    df = df[["url", "label"]].reset_index(drop=True)
    norm_counts = df["label"].value_counts().sort_index()
    print(f"\nNormalized label counts:")
    print(f"  0 (legitimate): {norm_counts.get(0, 0):,}")
    print(f"  1 (phishing):   {norm_counts.get(1, 0):,}")

    # ── Random stratified split (70 / 15 / 15) ────────────────────────────
    X = df["url"]
    y = df["label"]

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=0.15 / 0.85,
        stratify=y_tv,
        random_state=RANDOM_SEED,
    )

    train_df = pd.DataFrame({"url": X_train, "label": y_train})
    val_df = pd.DataFrame({"url": X_val, "label": y_val})
    test_df = pd.DataFrame({"url": X_test, "label": y_test})

    train_df.to_csv(PROCESSED / "urls_train.csv", index=False)
    val_df.to_csv(PROCESSED / "urls_val.csv", index=False)
    test_df.to_csv(PROCESSED / "urls_test.csv", index=False)

    # ── Domain-aware split ────────────────────────────────────────────────
    _build_domain_split(df)

    stats = {
        "random_seed": RANDOM_SEED,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "total_after_filtering": int(len(df)),
        "train": {
            "total": int(len(train_df)),
            "phishing": int((train_df["label"] == 1).sum()),
            "legitimate": int((train_df["label"] == 0).sum()),
        },
        "val": {
            "total": int(len(val_df)),
            "phishing": int((val_df["label"] == 1).sum()),
            "legitimate": int((val_df["label"] == 0).sum()),
        },
        "test_random": {
            "total": int(len(test_df)),
            "phishing": int((test_df["label"] == 1).sum()),
            "legitimate": int((test_df["label"] == 0).sum()),
        },
    }
    (PROCESSED / "urls_split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSplits saved to: {PROCESSED}")
    print(f"  Train       : {len(train_df):,}")
    print(f"  Val         : {len(val_df):,}")
    print(f"  Test(random): {len(test_df):,}")


def _build_domain_split(df: pd.DataFrame) -> None:
    """
    Attempt a domain-aware split where no registered domain appears in both
    training and test sets (reduces leakage from repeated domains).
    """
    try:
        import tldextract  # noqa: PLC0415
    except ImportError:
        print("[WARN] tldextract not available – skipping domain-aware split.")
        return

    print("\nBuilding domain-aware split …")
    domains = df["url"].apply(
        lambda u: tldextract.extract(u).top_domain_under_public_suffix or "__none__"
    )
    df = df.copy()
    df["domain"] = domains

    unique_domains = df["domain"].unique()
    from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    try:
        train_idx, test_idx = next(gss.split(df, df["label"], groups=df["domain"]))
    except Exception as exc:
        print(f"[WARN] Domain-aware split failed: {exc}")
        return

    train_domain = df.iloc[train_idx]
    test_domain = df.iloc[test_idx]

    # Verify no domain overlap
    train_domains = set(train_domain["domain"].unique())
    test_domains = set(test_domain["domain"].unique())
    overlap = train_domains & test_domains - {"__none__"}
    if overlap:
        print(f"  [WARN] {len(overlap)} overlapping domains – domain split may have limited value.")
    else:
        print(f"  ✓ No domain overlap between train and domain-test sets.")

    print(f"  Domain-aware train: {len(train_domain):,}, test: {len(test_domain):,}")

    train_domain[["url", "label"]].to_csv(
        PROCESSED / "urls_domain_train.csv", index=False
    )
    test_domain[["url", "label"]].to_csv(
        PROCESSED / "urls_domain_test.csv", index=False
    )
    print(f"  Saved domain-aware splits to {PROCESSED}")


if __name__ == "__main__":
    prepare()
