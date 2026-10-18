"""
prepare_messages.py
===================
Prepares the SMS/Smishing dataset for model training.

Steps:
  1. Load raw CSV with auto-detected column names.
  2. Map labels: ham → 0, smishing → 1. Exclude generic spam.
  3. Remove empty/invalid rows and exact duplicates.
  4. Stratified train / val / test split (70 / 15 / 15).
  5. Save splits + statistics to data/processed/.

Usage (from repo root):
    python src/fraud_phishing/data/prepare_messages.py
"""

import json
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_SMS = pathlib.Path("data/raw/SMS_dataset.csv")
PROCESSED = pathlib.Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── Label mapping ──────────────────────────────────────────────────────────
# Source labels (case-insensitive) → binary integer labels
LABEL_MAP = {
    "ham": 0,
    "smishing": 1,
}
EXCLUDED_LABELS = {"spam"}  # excluded from baseline binary experiment


def load_raw(path: pathlib.Path) -> pd.DataFrame:
    """Load raw CSV with multiple encoding attempts."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded {path.name} ({enc}): {len(df):,} rows, cols={list(df.columns)}")
            return df
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path}")


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect the label column and the text column.
    Returns (label_col, text_col).
    Raises ValueError if columns cannot be identified.
    """
    label_candidates = ["label", "type", "category", "class", "target"]
    text_candidates = ["text", "message", "msg", "sms", "body", "content"]

    col_lower = {c.lower(): c for c in df.columns}

    label_col = None
    for cand in label_candidates:
        if cand in col_lower:
            label_col = col_lower[cand]
            break

    text_col = None
    for cand in text_candidates:
        if cand in col_lower:
            text_col = col_lower[cand]
            break

    if label_col is None or text_col is None:
        raise ValueError(
            f"Could not detect label/text columns in: {list(df.columns)}. "
            "Update LABEL_MAP and column detection in prepare_messages.py."
        )

    print(f"Detected columns → label: '{label_col}', text: '{text_col}'")
    return label_col, text_col


def prepare(path: pathlib.Path = RAW_SMS) -> None:
    if not path.exists():
        print(f"[ERROR] Raw SMS dataset not found at: {path}")
        print("Run: python src/fraud_phishing/data/download_data.py")
        sys.exit(1)

    df = load_raw(path)
    label_col, text_col = detect_columns(df)

    # ── Normalise label strings ────────────────────────────────────────────
    df["label_str"] = df[label_col].astype(str).str.strip().str.lower()

    raw_counts = df["label_str"].value_counts()
    print(f"\nRaw label distribution:\n{raw_counts.to_string()}")

    # ── Log excluded categories ────────────────────────────────────────────
    excluded = df[df["label_str"].isin(EXCLUDED_LABELS)]
    print(
        f"\nExcluded '{','.join(EXCLUDED_LABELS)}' rows: {len(excluded):,} "
        f"({len(excluded)/len(df)*100:.1f}%)"
    )

    # ── Filter to known binary classes ────────────────────────────────────
    df = df[df["label_str"].isin(LABEL_MAP)].copy()
    df["label"] = df["label_str"].map(LABEL_MAP).astype(int)

    # ── Remove empty / whitespace-only messages ───────────────────────────
    pre_empty = len(df)
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""]
    df = df[df[text_col].str.strip().str.lower() != "nan"]
    print(f"\nRemoved empty messages: {pre_empty - len(df):,}")

    # ── Remove exact duplicates (same text AND label) ─────────────────────
    pre_dup = len(df)
    df = df.drop_duplicates(subset=[text_col, "label"])
    print(f"Removed exact duplicates: {pre_dup - len(df):,}")

    # ── Final counts ──────────────────────────────────────────────────────
    final_counts = df["label"].value_counts().sort_index()
    print(f"\nFinal binary label counts:\n  0 (ham):      {final_counts.get(0, 0):,}")
    print(f"  1 (smishing): {final_counts.get(1, 0):,}")
    ratio = final_counts.get(1, 0) / len(df)
    print(f"  Positive class ratio: {ratio:.3f}")

    # ── Stratified split: 70 / 15 / 15 ───────────────────────────────────
    df = df[[text_col, "label"]].rename(columns={text_col: "text"})

    X = df["text"]
    y = df["label"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.15 / 0.85,  # ~15% of total
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df = pd.DataFrame({"text": X_val, "label": y_val})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df.to_csv(PROCESSED / "messages_train.csv", index=False)
    val_df.to_csv(PROCESSED / "messages_val.csv", index=False)
    test_df.to_csv(PROCESSED / "messages_test.csv", index=False)

    stats = {
        "random_seed": RANDOM_SEED,
        "label_map": LABEL_MAP,
        "excluded_labels": list(EXCLUDED_LABELS),
        "excluded_count": int(len(excluded)),
        "total_after_filtering": int(len(df)),
        "train": {
            "total": int(len(train_df)),
            "positive": int((train_df["label"] == 1).sum()),
            "negative": int((train_df["label"] == 0).sum()),
        },
        "val": {
            "total": int(len(val_df)),
            "positive": int((val_df["label"] == 1).sum()),
            "negative": int((val_df["label"] == 0).sum()),
        },
        "test": {
            "total": int(len(test_df)),
            "positive": int((test_df["label"] == 1).sum()),
            "negative": int((test_df["label"] == 0).sum()),
        },
    }
    stats_path = PROCESSED / "messages_split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSplits saved to: {PROCESSED}")
    print(f"  Train : {len(train_df):,} ({train_df['label'].mean():.3f} positive)")
    print(f"  Val   : {len(val_df):,} ({val_df['label'].mean():.3f} positive)")
    print(f"  Test  : {len(test_df):,} ({test_df['label'].mean():.3f} positive)")
    print(f"  Stats : {stats_path}")


if __name__ == "__main__":
    prepare()
