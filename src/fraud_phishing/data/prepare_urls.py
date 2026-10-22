"""
prepare_urls.py
===============
Prepares the PHIUSIL raw URL dataset for model training.

Steps:
  1. Load raw CSV (produced by download_data.py).
  2. Verify and normalize labels: legitimate → 0, phishing → 1.
  3. Drop missing/empty URLs and exact duplicates.
  4. Stratified train / val / test split (70 / 15 / 15).
  5. Also build domain-aware and template-clean test splits, and report the
     leakage that each evaluation split still contains.
  6. Save splits to data/processed/.

Usage (from repo root):
    python src/fraud_phishing/data/prepare_urls.py
"""

import json
import pathlib
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# tldextract imported lazily (used only for domain grouping)

RAW_URL = pathlib.Path("data/raw/PHIUSIL_urls.csv")
PROCESSED = pathlib.Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── Label normalization ────────────────────────────────────────────────────
# AUTHORITATIVE RAW CONVENTION -- verified against the official UCI dataset
# card for id=967, which states verbatim:
#
#     "Class Labels: Label 1 corresponds to a legitimate URL,
#      label 0 to a phishing URL"
#
# and: "134,850 legitimate and 100,945 phishing URLs."
# Source: https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
#
# PHIUSIL therefore stores 1 = LEGITIMATE and 0 = PHISHING -- the opposite of
# the intuitive convention. Reading it the other way inverts every metric,
# every confusion matrix and every label the API returns, so the raw
# composition is asserted in prepare() below and the pipeline fails loudly
# rather than silently reporting inverted results.
#
# This project's convention (applied to every downstream artefact):
#     raw 1 (legitimate) -> 0   negative class
#     raw 0 (phishing)   -> 1   positive class
PHIUSIL_LABEL_CONVENTION_SOURCE = (
    "https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset"
)
PHIUSIL_RAW_LABEL_MEANING = {1: "legitimate", 0: "phishing"}
PHIUSIL_EXPECTED_RAW_COUNTS = {1: 134_850, 0: 100_945}

PHIUSIL_LABEL_MAP_STR = {
    "1": 0,   # raw 1 = legitimate -> project 0
    "0": 1,   # raw 0 = phishing   -> project 1
    "-1": 0,  # legacy exports occasionally use -1 for legitimate
}
PHIUSIL_LABEL_MAP_INT = {
    1: 0,
    0: 1,
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

    def _map(v):
        sv = str(v).strip()
        if sv in str_map:
            return str_map[sv]
        raise ValueError(f"Unknown label value: {v!r}")

    return series.apply(_map).astype(int)


def verify_raw_label_convention(raw_counts: pd.Series) -> None:
    """
    Fail loudly unless the raw label counts match the composition documented on
    the official UCI dataset card (1 = legitimate, 0 = phishing).

    Reading PHIUSIL's labels the other way round silently inverts every metric
    and every label the API returns, so this is a hard error rather than a
    warning.
    """
    observed: dict[int, int] = {}
    for key, value in raw_counts.items():
        try:
            observed[int(key)] = int(value)
        except (TypeError, ValueError):
            continue

    mismatch = {
        raw: (observed.get(raw), expected)
        for raw, expected in PHIUSIL_EXPECTED_RAW_COUNTS.items()
        if observed.get(raw) != expected
    }
    if mismatch:
        detail = ", ".join(
            f"raw {raw}: expected {exp:,}, found {got if got is None else format(got, ',')}"
            for raw, (got, exp) in mismatch.items()
        )
        raise ValueError(
            "PHIUSIL raw label distribution does not match the documented "
            f"composition ({detail}).\n"
            "The official dataset card states: 'Label 1 corresponds to a "
            "legitimate URL, label 0 to a phishing URL'.\n"
            f"Verify the convention at {PHIUSIL_LABEL_CONVENTION_SOURCE} "
            "before trusting any metric produced from this file."
        )

    print(
        "  [OK] Raw label convention verified -> "
        f"1 = legitimate ({observed[1]:,}), 0 = phishing ({observed[0]:,})"
    )


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
    verify_raw_label_convention(raw_counts)

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
    print(f"\nNormalized label counts (project convention):")
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

    # ── Assert the random splits are disjoint on exact URL ────────────────
    split_sets = {
        "train": set(train_df["url"]),
        "val": set(val_df["url"]),
        "test": set(test_df["url"]),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        dup = split_sets[a] & split_sets[b]
        if dup:
            raise RuntimeError(
                f"Exact-URL overlap between {a} and {b}: {len(dup):,} URLs. "
                "The splits are not disjoint."
            )
    print("\n[OK] Random splits are disjoint on exact URL.")

    # ── Domain-aware + template-clean splits, with a leakage report ───────
    leakage = _build_domain_split(df, train_df, val_df)

    stats = {
        "random_seed": RANDOM_SEED,
        "label_map": {"0": "legitimate", "1": "phishing"},
        "label_convention": {
            "note": (
                "PHIUSIL raw labels are 1 = legitimate, 0 = phishing. They are "
                "mapped to the project convention 0 = legitimate, 1 = phishing."
            ),
            "raw_meaning": {str(k): v for k, v in PHIUSIL_RAW_LABEL_MEANING.items()},
            "expected_raw_counts": {
                str(k): v for k, v in PHIUSIL_EXPECTED_RAW_COUNTS.items()
            },
            "source": PHIUSIL_LABEL_CONVENTION_SOURCE,
        },
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
        "leakage": leakage,
    }
    (PROCESSED / "urls_split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSplits saved to: {PROCESSED}")
    print(f"  Train       : {len(train_df):,}")
    print(f"  Val         : {len(val_df):,}")
    print(f"  Test(random): {len(test_df):,}")


def url_template(url: str) -> str:
    """
    Normalized 'template' used for near-duplicate detection.

    Lowercases, strips the scheme, drops the query/fragment, removes a trailing
    slash and masks digit runs, so structurally identical pages that differ only
    by host or by an id/number collapse to the same key.
    """
    u = url.strip().lower()
    u = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", u)
    u = u.split("#", 1)[0].split("?", 1)[0]
    u = u.rstrip("/")
    return re.sub(r"\d+", "#", u)


def leakage_report(
    train_urls: list[str],
    test_urls: list[str],
    domains: dict[str, str],
    label: str,
) -> dict:
    """Quantify how much of a test set is also present in a training set."""
    train_set, test_set = set(train_urls), set(test_urls)
    train_domains = {domains[u] for u in train_set if domains.get(u, "__none__") != "__none__"}
    test_domains = {domains[u] for u in test_set if domains.get(u, "__none__") != "__none__"}
    train_templates = {url_template(u) for u in train_set}
    test_templates = [url_template(u) for u in test_set]
    template_hits = sum(1 for t in test_templates if t in train_templates)

    return {
        "split": label,
        "n_train": len(train_set),
        "n_test": len(test_set),
        "exact_url_overlap": len(train_set & test_set),
        "domain_overlap": len(train_domains & test_domains),
        "test_template_overlap_count": template_hits,
        "test_template_overlap_pct": round(
            template_hits / max(len(test_set), 1) * 100, 2
        ),
    }


def _build_domain_split(
    df: pd.DataFrame,
    random_train: pd.DataFrame | None = None,
    random_val: pd.DataFrame | None = None,
) -> dict | None:
    """
    Build a domain-aware split (no registered domain shared between train and
    test) plus a template-clean view of the test set, and report the leakage
    that each evaluation split still contains.

    Returns the leakage report dict, or None if the split could not be built.
    """
    try:
        import tldextract  # noqa: PLC0415
    except ImportError:
        print("[WARN] tldextract not available - skipping domain-aware split.")
        return None

    print("\nBuilding domain-aware + template-clean splits ...")
    df = df.copy()
    df["domain"] = df["url"].apply(
        lambda u: tldextract.extract(u).top_domain_under_public_suffix or "__none__"
    )
    domains = dict(zip(df["url"], df["domain"]))

    from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    try:
        train_idx, test_idx = next(gss.split(df, df["label"], groups=df["domain"]))
    except Exception as exc:
        print(f"[WARN] Domain-aware split failed: {exc}")
        return None

    train_domain = df.iloc[train_idx]
    test_domain = df.iloc[test_idx]

    # ── Domain isolation is a hard requirement, not a warning ─────────────
    train_domains = set(train_domain["domain"]) - {"__none__"}
    test_domains = set(test_domain["domain"]) - {"__none__"}
    overlap = train_domains & test_domains
    if overlap:
        raise RuntimeError(
            f"Domain-aware split has {len(overlap)} overlapping domains; "
            "the split is not domain-isolated. Check RANDOM_SEED / GroupShuffleSplit."
        )
    print("  [OK] No registered domain appears in both sides of the domain split.")

    # ── Template-clean test set ───────────────────────────────────────────
    # Domain isolation alone is not enough: PHIUSIL contains many URLs sharing a
    # masked path template across different hosts (e.g. '<name>-<id>.weeblysite.com'),
    # which are near-duplicates of training rows.
    train_templates = {url_template(u) for u in train_domain["url"]}
    keep_mask = test_domain["url"].map(lambda u: url_template(u) not in train_templates)
    test_template = test_domain[keep_mask]
    dropped = int((~keep_mask).sum())

    train_domain[["url", "label"]].to_csv(PROCESSED / "urls_domain_train.csv", index=False)
    test_domain[["url", "label"]].to_csv(PROCESSED / "urls_domain_test.csv", index=False)
    train_domain[["url", "label"]].to_csv(PROCESSED / "urls_template_train.csv", index=False)
    test_template[["url", "label"]].to_csv(PROCESSED / "urls_template_test.csv", index=False)

    print(f"  Domain-aware train: {len(train_domain):,}  test: {len(test_domain):,}")
    print(
        f"  Template-clean test: {len(test_template):,} "
        f"({dropped:,} rows removed as near-duplicates of training rows)"
    )

    # ── Leakage report ────────────────────────────────────────────────────
    reports = [
        leakage_report(
            list(train_domain["url"]),
            list(test_domain["url"]),
            domains,
            "domain_train vs domain_test",
        ),
        leakage_report(
            list(train_domain["url"]),
            list(test_template["url"]),
            domains,
            "domain_train vs template_clean_test",
        ),
    ]
    if random_train is not None and random_val is not None:
        reports.insert(
            0,
            leakage_report(
                list(random_train["url"]) + list(random_val["url"]),
                list(test_domain["url"]),
                domains,
                "LEGACY random-train+val vs domain_test (contaminated)",
            ),
        )

    print("\n  Leakage report:")
    for rep in reports:
        print(
            f"    {rep['split']:<52s} url={rep['exact_url_overlap']:>6,} "
            f"domain={rep['domain_overlap']:>5,} "
            f"template={rep['test_template_overlap_pct']:>5.1f}%"
        )

    report_path = pathlib.Path("reports") / "url_leakage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"random_seed": RANDOM_SEED, "reports": reports}, indent=2),
        encoding="utf-8",
    )
    print(f"  Leakage report written to: {report_path}")

    return {
        "random_seed": RANDOM_SEED,
        "domain_train_total": int(len(train_domain)),
        "domain_test_total": int(len(test_domain)),
        "template_test_total": int(len(test_template)),
        "template_test_rows_removed": dropped,
        "reports": reports,
    }


if __name__ == "__main__":
    prepare()
