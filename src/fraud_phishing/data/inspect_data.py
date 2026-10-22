"""
inspect_data.py
===============
Inspects the raw downloaded datasets and produces reports/dataset_inventory.md.

Usage (from repo root):
    python src/fraud_phishing/data/inspect_data.py
"""

import hashlib
import pathlib
import sys
import textwrap

import pandas as pd

RAW_DIR = pathlib.Path("data/raw")
REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SMS_PATH = RAW_DIR / "SMS_dataset.csv"
URL_PATH = RAW_DIR / "PHIUSIL_urls.csv"
LEGITPHISH_PATH = RAW_DIR / "LegitPhish_urls.csv"
MURL_PATH = RAW_DIR / "MURL_urls.csv"
URLPHISH_PATH = RAW_DIR / "URLPhish_urls.csv"


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
            "- **Raw convention (official UCI dataset card):**",
            "  `1` = legitimate, `0` = phishing —",
            '  *"Class Labels: Label 1 corresponds to a legitimate URL, '
            'label 0 to a phishing URL"*',
            "- **Project convention applied downstream:** legitimate -> 0, phishing -> 1",
            "- Expected raw composition: 134,850 legitimate / 100,945 phishing",
            "- Source: https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset",
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
# Dataset C – LegitPhish (external evaluation only)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_legitphish() -> str:
    lines = ["## Dataset C - LegitPhish URLs (development benchmark)\n"]

    if not LEGITPHISH_PATH.exists():
        lines.append(
            f"- **Not present.** Expected at `{LEGITPHISH_PATH}`.\n"
            "- Run `python src/fraud_phishing/data/download_data.py` to fetch it."
        )
        return "\n".join(lines)

    lines += [
        "- **Source:** https://data.mendeley.com/datasets/hx4m73v2sf/2",
        "- **Citation:** Potpelwar, R., Kulkarni, U. & Waghmare, J. (2025). "
        "*LegitPhish Dataset.* Mendeley Data, V2. https://doi.org/10.17632/hx4m73v2sf.2",
        "- **License:** CC BY 4.0",
        "- **Role:** development / diagnostic benchmark. No model is trained on it, "
        "but feature and model choices were informed by results on it, so it is no "
        "longer a clean generalisation claim. Dataset E fills that role.",
        "- **Provenance:** phishing URLs sourced from URLHaus and other threat "
        "repositories; legitimate URLs scraped from Wikipedia, Stack Overflow and "
        "high-authority sites.",
        "- **Note:** its legitimate URLs are byte-identical to PHIUSIL's except for "
        "445 URLs, so a precision figure computed on this class is mostly a "
        "memorisation check. Those 445 URLs are the ones the P3 diagnosis uses.",
    ]

    digest = hashlib.sha256(LEGITPHISH_PATH.read_bytes()).hexdigest()
    lines.append(f"- **SHA-256 (as stored):** `{digest}`")

    df = load_csv_safe(LEGITPHISH_PATH)
    if df is None:
        return "\n".join(lines)

    info = describe_df(df, "LegitPhish")
    lines += [
        f"- **Rows:** {info['rows']:,}",
        f"- **Columns:** {len(info['columns'])}",
        f"- **Exact duplicate rows:** {info['exact_duplicates']:,}",
    ]

    label_col = "ClassLabel" if "ClassLabel" in df.columns else _find_label_col(df)
    if label_col:
        vc = df[label_col].dropna().value_counts().sort_index()
        lines.append("\n### Raw label distribution")
        total = int(vc.sum())
        for val, cnt in vc.items():
            lines.append(f"  - `{int(val)}`: {int(cnt):,} ({int(cnt) / total * 100:.1f}%)")
        lines.append("\n### Label normalization policy")
        lines += [
            "- **Raw convention (dataset card + verified against observed domains):**",
            "  `1` = legitimate, `0` = phishing - the *same inverted convention as PHIUSIL*",
            "- **Project convention applied downstream:** legitimate -> 0, phishing -> 1",
            "- Expected raw composition: 63,678 phishing / 37,540 legitimate",
            "- `prepare_external_urls.py` fails hard if these counts ever change.",
        ]

    url_col = _find_url_col(df)
    if url_col:
        dupes = df[url_col].duplicated().sum()
        lines.append(f"\n**Duplicate URL strings:** {dupes:,}")

    # Overlap with PHIUSIL - the key caveat about this dataset's independence.
    if URL_PATH.exists():
        phiusil = load_csv_safe(URL_PATH)
        if phiusil is not None:
            pcol = _find_url_col(phiusil)
            if pcol and url_col:
                pool = set(phiusil[pcol].astype(str))
                urls = df[url_col].astype(str)
                identical = int(urls.isin(pool).sum())
                lines.append("\n### Overlap with PHIUSIL")
                lines.append(
                    f"- URLs byte-identical to a PHIUSIL URL: "
                    f"**{identical:,} / {len(df):,} ({identical / len(df) * 100:.2f}%)**"
                )
                if label_col:
                    for raw, name in ((1, "legitimate"), (0, "phishing")):
                        sub = df[df[label_col] == raw]
                        if not len(sub):
                            continue
                        ident = int(sub[url_col].astype(str).isin(pool).sum())
                        lines.append(
                            f"  - {name}: {ident:,} / {len(sub):,} "
                            f"({ident / len(sub) * 100:.2f}%) byte-identical"
                        )
                lines.append(
                    "- **Caveat:** nearly all of LegitPhish's *legitimate* URLs are byte-"
                    "identical to PHIUSIL's, so precision measured on that class is "
                    "optimistic. Its *phishing* class has zero PHIUSIL overlap and is a "
                    "genuine generalisation test. See `reports/external_evaluation.md`."
                )

    return "\n".join(lines)


def inspect_murl() -> str:
    lines = ["## Dataset D - MURL URLs (training augmentation only)\n"]

    if not MURL_PATH.exists():
        lines.append(
            f"- **Not present.** Expected at `{MURL_PATH}`.\n"
            "- Run `python src/fraud_phishing/data/download_data.py` to fetch it."
        )
        return "\n".join(lines)

    lines += [
        "- **Source:** https://data.mendeley.com/datasets/5fk24y4xmg/2",
        "- **License:** CC BY 4.0",
        "- **Role:** training augmentation only - it supplies path-bearing, "
        "scheme-less URLs that PHIUSIL lacks. It is never used for evaluation.",
        "- **Caveat:** URLs are stored **without a scheme**. The extractor's "
        "`is_https` therefore reads 0 for every MURL row, which is a property of "
        "the file, not of the URL. See `reports/limitations.md`.",
    ]

    digest = hashlib.sha256(MURL_PATH.read_bytes()).hexdigest()
    lines.append(f"- **SHA-256 (as stored):** `{digest}`")

    df = load_csv_safe(MURL_PATH)
    if df is None:
        return "\n".join(lines)

    info = describe_df(df, "MURL")
    lines += [
        f"- **Rows:** {info['rows']:,}",
        f"- **Columns:** {len(info['columns'])}",
        f"- **Exact duplicate rows:** {info['exact_duplicates']:,}",
    ]

    if "type" in df.columns:
        lines.append("\n### Raw label distribution (`type`)")
        vc = df["type"].astype(str).str.strip().str.lower().value_counts()
        total = int(vc.sum())
        for val, cnt in vc.items():
            lines.append(f"  - `{val}`: {int(cnt):,} ({int(cnt) / total * 100:.1f}%)")
        lines += [
            "\n### Label normalization policy",
            "- **Project convention:** `benign` and `legitimate` -> 0, "
            "`phishing` -> 1",
            "- All other categories (`defacement`, `malware`, `spam`, `Exploit`) "
            "are dropped rather than folded into either class.",
            "- Benign and phishing are capped at 60,000 rows each with a fixed "
            "seed so neither class dominates the augmented pool.",
        ]

    if "url" in df.columns:
        dupes = df["url"].duplicated().sum()
        lines.append(f"\n**Duplicate URL strings:** {dupes:,}")
        no_scheme = int(
            (~df["url"].astype(str).str.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")).sum()
        )
        lines.append(
            f"- URLs without an explicit scheme: **{no_scheme:,} / {len(df):,} "
            f"({no_scheme / len(df) * 100:.1f}%)**"
        )

    return "\n".join(lines)


def inspect_urlphish() -> str:
    lines = ["## Dataset E - URL-Phish (final untouched external test)\n"]

    if not URLPHISH_PATH.exists():
        lines.append(
            f"- **Not present.** Expected at `{URLPHISH_PATH}`.\n"
            "- Run `python src/fraud_phishing/data/download_data.py` to fetch it."
        )
        return "\n".join(lines)

    lines += [
        "- **Source:** https://data.mendeley.com/datasets/65z9twcx3r/2",
        "- **License:** CC BY 4.0",
        "- **Role:** the **final, untouched external test set**. No feature "
        "selection, threshold choice or model choice in this project used it.",
        "- **Provenance:** phishing sample from PhishTank; benign sample from the "
        "Research Organization Registry (ROR) list of legitimate institutions.",
        "- **Why it matters:** its HTTPS rate is *inverted* relative to PHIUSIL "
        "(most phishing is HTTPS, much benign traffic is plain HTTP), so a model "
        "that learned \"https implies legitimate\" cannot pass it.",
    ]

    digest = hashlib.sha256(URLPHISH_PATH.read_bytes()).hexdigest()
    lines.append(f"- **SHA-256 (as stored):** `{digest}`")

    df = load_csv_safe(URLPHISH_PATH)
    if df is None:
        return "\n".join(lines)

    info = describe_df(df, "URL-Phish")
    lines += [
        f"- **Rows:** {info['rows']:,}",
        f"- **Columns:** {len(info['columns'])}",
    ]

    label_col = "label" if "label" in df.columns else _find_label_col(df)
    if label_col:
        vc = df[label_col].dropna().value_counts().sort_index()
        lines.append("\n### Raw label distribution")
        total = int(vc.sum())
        for val, cnt in vc.items():
            lines.append(f"  - `{int(val)}`: {int(cnt):,} ({int(cnt) / total * 100:.1f}%)")
        lines += [
            "\n### Label normalization policy",
            "- **Raw convention:** `0` = benign, `1` = phishing - the *intuitive* "
            "convention, unlike PHIUSIL and LegitPhish.",
            "- This already matches the project convention, so labels are used as-is.",
            "- `prepare_external_urls.py` fails hard if the composition ever changes.",
        ]

    url_col = _find_url_col(df)
    if url_col:
        dupes = df[url_col].duplicated().sum()
        lines.append(f"\n**Duplicate URL strings:** {dupes:,}")

    stats_path = REPORTS_DIR / "urlphish_external_stats.json"
    if stats_path.exists():
        import json

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        rows = stats.get("rows", {})
        overlap = stats.get("overlap", {})
        lines.append(
            "\n### Prepared evaluation split (`data/processed/urlphish_external.csv`)"
        )
        lines.append(
            f"- Rows kept after de-duplication: **{rows.get('after_dedup', 0):,}** "
            f"({rows.get('benign', 0):,} benign / {rows.get('phishing', 0):,} phishing)"
        )
        identical = int(overlap.get("byte_identical_to_any_training", 0))
        lines.append(
            f"- Byte-identical to a training URL: **{identical:,}** "
            "(all benign; these are excluded from the headline generalisation "
            "metrics and reported separately as an integrity stratum)"
        )
        lines.append(
            f"- Rows whose registered domain appears in PHIUSIL training data: "
            f"**{int(overlap.get('rows_on_phiusil_training_domain', 0)):,}**"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Inspecting datasets …\n")

    sms_section = inspect_sms()
    url_section = inspect_urls()
    legitphish_section = inspect_legitphish()
    murl_section = inspect_murl()
    urlphish_section = inspect_urlphish()

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
    inventory += legitphish_section
    inventory += "\n\n---\n\n"
    inventory += murl_section
    inventory += "\n\n---\n\n"
    inventory += urlphish_section
    inventory += "\n\n---\n\n"
    inventory += textwrap.dedent(
        """\
        ## Dataset roles at a glance
        | ID | Dataset | Role |
        |----|---------|------|
        | A | Mendeley SMS/Smishing | training + in-distribution test |
        | B | PHIUSIL URLs | training (random and domain-held-out splits) |
        | C | LegitPhish URLs | development / diagnostic benchmark only |
        | D | MURL URLs | training augmentation only |
        | E | URL-Phish URLs | **final untouched external test** |

        ## Usage Notes
        - Raw files must remain unchanged under `data/raw/`.
        - Processed splits are saved to `data/processed/` by the prepare scripts.
        - Dataset C is a *development* benchmark: it informed feature and model
          choices, so results on it are not a clean generalisation claim.
          Dataset E was never used for any decision and is the only clean claim.
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
    print()
    print(legitphish_section)
    print()
    print(murl_section)
    print()
    print(urlphish_section)


if __name__ == "__main__":
    main()
