"""
inspect_data.py
===============
Inspects the raw downloaded datasets and produces reports/dataset_inventory.md.

Usage (from repo root):
    python src/fraud_phishing/data/inspect_data.py
"""

import pathlib
import sys
import textwrap

import pandas as pd

RAW_DIR = pathlib.Path("data/raw")
REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SMS_PATH = RAW_DIR / "SMS_dataset.csv"
URL_PATH = RAW_DIR / "PHIUSIL_urls.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_safe(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
    """Try common encodings to load a CSV."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            print(f"  Loaded {path.name} with encoding={enc}")
            return df
        except Exception:
            continue
    print(f"  [ERROR] Could not load {path.name}")
    return None


def describe_df(df: pd.DataFrame, name: str) -> dict:
    info = {}
    info["rows"] = len(df)
    info["columns"] = list(df.columns)
    info["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    info["nulls"] = df.isnull().sum().to_dict()
    info["exact_duplicates"] = int(df.duplicated().sum())
    # Per-column value counts for object columns
    info["value_counts"] = {}
    for c in df.select_dtypes("object").columns:
        info["value_counts"][c] = df[c].value_counts(dropna=False).head(20).to_dict()
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Dataset A – SMS / Smishing
# ─────────────────────────────────────────────────────────────────────────────

def inspect_sms() -> str:
    lines = ["## Dataset A – SMS / Smishing Messages\n"]
    lines += [
        "- **Source:** https://data.mendeley.com/datasets/f45bkkt8pr/1",
        "- **License:** CC BY 4.0",
        "- **Citation:** Mishra, S. & Soni, D. (2019).",
        "  *SMS Phishing Dataset for Machine Learning and Pattern Recognition.*",
        "  Mendeley Data, V1.",
    ]

    if not SMS_PATH.exists():
        lines.append(
            "\n> **⚠ File not found.**  "
            "Follow manual download instructions in `src/fraud_phishing/data/download_data.py`."
        )
        return "\n".join(lines)

    df = load_csv_safe(SMS_PATH)
    if df is None:
        lines.append("\n> **⚠ Could not parse file.**")
        return "\n".join(lines)

    info = describe_df(df, "SMS")

    lines += [
        f"\n**File:** `{SMS_PATH}`",
        f"**Rows:** {info['rows']:,}",
        f"**Columns:** {info['columns']}",
        "",
        "### Column dtypes",
    ]
    for col, dtype in info["dtypes"].items():
        lines.append(f"- `{col}`: {dtype}")

    lines.append("\n### Missing values")
    for col, n in info["nulls"].items():
        lines.append(f"- `{col}`: {n}")

    lines.append(f"\n**Exact duplicate rows:** {info['exact_duplicates']:,}")

    lines.append("\n### Label distribution (all label-like columns)")
    for col, vc in info["value_counts"].items():
        if col.lower() in ("label", "type", "category", "class", "target"):
            lines.append(f"\n**Column `{col}`:**")
            total = sum(vc.values())
            for val, cnt in sorted(vc.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100
                lines.append(f"  - `{val}`: {cnt:,} ({pct:.1f}%)")

    lines.append("\n### Binary classification policy")
    lines += [
        "- **ham** → label 0 (negative / legitimate)",
        "- **smishing** → label 1 (positive / phishing)",
        "- **spam** → **excluded** from baseline binary experiment",
        "  (generic spam is not verified to match the phishing positive class).",
    ]

    # Empty / whitespace-only messages
    text_col = _find_text_col(df)
    if text_col:
        empty = df[text_col].isna().sum() + (df[text_col].str.strip() == "").sum()
        lines.append(f"\n**Empty or whitespace-only messages** (column `{text_col}`): {empty}")

    return "\n".join(lines)


def _find_text_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ("text", "message", "msg", "sms", "body", "content"):
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset B – PHIUSIL URLs
# ─────────────────────────────────────────────────────────────────────────────

def inspect_urls() -> str:
    lines = ["## Dataset B – PHIUSIL Phishing URL Dataset\n"]
    lines += [
        "- **Source:** https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset",
        "- **License:** CC BY 4.0",
        "- **Citation:** Hannousse, A. (2023).",
        "  *PHIUSIL: A phishing URL dataset.* UCI Machine Learning Repository.",
        "  https://doi.org/10.24432/C58S7F",
    ]

    if not URL_PATH.exists():
        lines.append(
            "\n> **⚠ File not found.**  "
            "Run `python src/fraud_phishing/data/download_data.py` or follow manual instructions."
        )
        return "\n".join(lines)

    df = load_csv_safe(URL_PATH)
    if df is None:
        lines.append("\n> **⚠ Could not parse file.**")
        return "\n".join(lines)

    info = describe_df(df, "PHIUSIL")

    lines += [
        f"\n**File:** `{URL_PATH}`",
        f"**Rows:** {info['rows']:,}",
        f"**Columns:** {info['columns']}",
        "",
        "### Column dtypes",
    ]
    for col, dtype in info["dtypes"].items():
        lines.append(f"- `{col}`: {dtype}")

    lines.append("\n### Missing values")
    for col, n in info["nulls"].items():
        lines.append(f"- `{col}`: {n}")

    lines.append(f"\n**Exact duplicate rows:** {info['exact_duplicates']:,}")

    lines.append("\n### Label distribution")
    label_col = _find_label_col(df)
    if label_col:
        vc = df[label_col].value_counts(dropna=False)
        total = len(df)
        for val, cnt in vc.items():
            pct = cnt / total * 100
            lines.append(f"  - `{val}`: {cnt:,} ({pct:.1f}%)")
        lines.append("\n### Label normalization policy")
        lines += [
            "- Legitimate URLs → 0",
            "- Phishing URLs → 1",
            "- Confirm mapping against dataset documentation before applying.",
        ]

    # Empty URLs
    url_col = _find_url_col(df)
    if url_col:
        empty = df[url_col].isna().sum() + (df[url_col].str.strip() == "").sum()
        lines.append(f"\n**Empty or missing URL strings** (column `{url_col}`): {empty}")
        dupes = df[url_col].duplicated().sum()
        lines.append(f"**Duplicate URL strings:** {dupes:,}")

    return "\n".join(lines)


def _find_label_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ("label", "label_raw", "target", "class", "phishing"):
            return c
    return None


def _find_url_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ("url", "urls", "link"):
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Inspecting datasets …\n")

    sms_section = inspect_sms()
    url_section = inspect_urls()

    inventory = textwrap.dedent(
        """\
        # Dataset Inventory
        **Project:** AI-Based Fraud and Phishing Detection with Explainable AI
        **Generated by:** `src/fraud_phishing/data/inspect_data.py`

        ---

        """
    )
    inventory += sms_section
    inventory += "\n\n---\n\n"
    inventory += url_section
    inventory += "\n\n---\n\n"
    inventory += textwrap.dedent(
        """\
        ## Usage Notes
        - Raw files must remain unchanged under `data/raw/`.
        - Processed splits are saved to `data/processed/` by the prepare scripts.
        - Do not commit raw data files to Git (see `.gitignore`).
        - Re-run this script after downloading datasets to update figures.
        """
    )

    out_path = REPORTS_DIR / "dataset_inventory.md"
    out_path.write_text(inventory, encoding="utf-8")
    print(f"\nDataset inventory written to: {out_path}")
    print(sms_section)
    print()
    print(url_section)


if __name__ == "__main__":
    main()
