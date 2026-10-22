"""
prepare_external_urls.py
========================
Prepares the LegitPhish dataset as an **external** URL evaluation set.

LegitPhish (Potpelwar et al., 2025) contains 101,219 URLs from a completely
different collection pipeline than PHIUSIL: its phishing URLs come from URLHaus /
PhishTank and its legitimate URLs were scraped from Wikipedia, Stack Overflow and
other high-authority domains. That makes it the right instrument for asking the
question PHIUSIL cannot answer: *does the model still work on data it has never
seen from a different source?*

This module deliberately does **not** produce training splits. LegitPhish is
evaluation-only until an independent baseline has been established.

Label convention (verified against the dataset card *and* the observed top
domains, which are wikipedia.org / senate.gov / af.mil for the legitimate class):

    raw ClassLabel 0 = PHISHING  -> project 1
    raw ClassLabel 1 = LEGITIMATE -> project 0

Note this is the same inverted convention as PHIUSIL.

Steps:
  1. Load data/raw/LegitPhish_urls.csv (produced by download_data.py).
  2. Verify the raw label composition against the dataset card.
  3. Normalize labels to the project convention (0 = legitimate, 1 = phishing).
  4. Drop missing/empty URLs and exact duplicates.
  5. Tag each URL with whether its registered domain ever appears in the
     PHIUSIL pools, so evaluation can separate "unseen domain" from "domain the
     model was actually trained on".
  6. Save to data/processed/legitphish_external.csv.

Usage (from repo root):
    python src/fraud_phishing/data/prepare_external_urls.py
"""

import json
import pathlib
import sys

import pandas as pd

RAW = pathlib.Path("data/raw/LegitPhish_urls.csv")
PROCESSED = pathlib.Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = pathlib.Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

OUT_CSV = PROCESSED / "legitphish_external.csv"
OUT_STATS = REPORTS / "legitphish_external_stats.json"

# ── Raw label convention (verified) ──────────────────────────────────────────
LEGITPHISH_LABEL_CONVENTION_SOURCE = "https://data.mendeley.com/datasets/hx4m73v2sf/2"
LEGITPHISH_RAW_LABEL_MEANING = {1: "legitimate", 0: "phishing"}
LEGITPHISH_EXPECTED_RAW_COUNTS = {1: 37_540, 0: 63_678}  # 1=legit, 0=phishing
LEGITPHISH_LABEL_MAP = {
    "1": 0,  # raw 1 = legitimate -> project 0
    "0": 1,  # raw 0 = phishing   -> project 1
}

# PHIUSIL pools whose registered domains must be treated as "seen in training".
PHIUSIL_POOL_FILES = ("urls_train.csv", "urls_val.csv", "urls_template_train.csv")


def load_raw(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] LegitPhish raw file not found at: {path}")
        print("Run: python src/fraud_phishing/data/download_data.py")
        sys.exit(1)

    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded {path.name} (encoding={enc}): {len(df):,} rows, cols={list(df.columns)}")
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def verify_raw_label_convention(raw_counts: pd.Series) -> None:
    """
    Fail loudly unless the raw ClassLabel counts match the dataset card.

    Reading LegitPhish's labels the other way round would invert the external
    evaluation completely, so this is a hard error.
    """
    observed: dict[int, int] = {}
    for key, value in raw_counts.items():
        try:
            observed[int(key)] = int(value)
        except (TypeError, ValueError):
            continue

    mismatch = {
        raw: (observed.get(raw), expected)
        for raw, expected in LEGITPHISH_EXPECTED_RAW_COUNTS.items()
        if observed.get(raw) != expected
    }
    if mismatch:
        detail = ", ".join(
            f"raw {raw}: expected {exp:,}, found "
            f"{got if got is None else format(got, ',')}"
            for raw, (got, exp) in mismatch.items()
        )
        raise ValueError(
            "LegitPhish raw label distribution does not match the dataset card "
            f"({detail}).\n"
            "The card states 'Target label: 1 = Legitimate, 0 = Phishing'. "
            "Verify the convention at "
            f"{LEGITPHISH_LABEL_CONVENTION_SOURCE} before trusting any metric."
        )

    print(
        "  [OK] Raw label convention verified -> "
        f"1 = legitimate ({observed[1]:,}), 0 = phishing ({observed[0]:,})"
    )


def normalize_label(series: pd.Series) -> pd.Series:
    """Map raw ClassLabel values to the project convention {0, 1}."""
    def _map(value):
        key = str(value).strip()
        if key.endswith(".0"):  # the raw file stores labels as floats
            key = key[:-2]
        if key in LEGITPHISH_LABEL_MAP:
            return LEGITPHISH_LABEL_MAP[key]
        raise ValueError(f"Unknown ClassLabel value: {value!r}")

    return series.apply(_map).astype(int)


def load_phiusil_domains() -> tuple[set[str], list[str]]:
    """Registered domains that appear in any PHIUSIL pool we train on."""
    try:
        import tldextract  # noqa: PLC0415
    except ImportError:
        print("[WARN] tldextract not available - cannot compute domain overlap.")
        return set(), []

    domains: set[str] = set()
    used: list[str] = []
    for name in PHIUSIL_POOL_FILES:
        path = PROCESSED / name
        if not path.exists():
            continue
        used.append(name)
        urls = pd.read_csv(path)["url"].astype(str)
        for url in urls:
            sub = tldextract.extract(url)
            domain = sub.top_domain_under_public_suffix
            if domain:
                domains.add(domain)
    return domains, used


def prepare() -> None:
    print("=" * 60)
    print("Preparing LegitPhish as an EXTERNAL evaluation set")
    print("=" * 60)

    df = load_raw(RAW)

    col_lower = {c.lower().strip(): c for c in df.columns}
    url_col = col_lower.get("url")
    label_col = col_lower.get("classlabel")
    if url_col is None or label_col is None:
        raise ValueError(
            f"Cannot find URL / ClassLabel columns in: {list(df.columns)}"
        )
    print(f"URL column: '{url_col}', Label column: '{label_col}'")

    raw_counts = df[label_col].dropna().value_counts().sort_index()
    print(f"\nRaw ClassLabel distribution:\n{raw_counts.to_string()}")
    verify_raw_label_convention(raw_counts)

    missing_labels = int(df[label_col].isna().sum())
    if missing_labels:
        print(f"  [NOTE] {missing_labels} row(s) have no label and are dropped.")

    df = df.dropna(subset=[label_col]).copy()
    df["label"] = normalize_label(df[label_col])
    df["url"] = df[url_col].astype(str).str.strip()

    before = len(df)
    df = df[df["url"].str.len() > 0]
    print(f"\nDropped empty URLs: {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    print(f"Dropped exact duplicate URLs: {before - len(df):,}")

    norm_counts = df["label"].value_counts().sort_index()
    print("\nNormalized label counts (project convention):")
    print(f"  0 (legitimate): {norm_counts.get(0, 0):,}")
    print(f"  1 (phishing):   {norm_counts.get(1, 0):,}")

    # ── Domain tag vs the PHIUSIL training pools ──────────────────────────────
    phiusil_domains, pools_used = load_phiusil_domains()
    print(f"\nPHIUSIL training pools considered: {pools_used or 'none found'}")
    print(f"Distinct PHIUSIL training domains: {len(phiusil_domains):,}")

    if phiusil_domains:
        import tldextract  # noqa: PLC0415

        def _domain(url: str) -> str:
            return tldextract.extract(url).top_domain_under_public_suffix or ""

        df["domain"] = df["url"].map(_domain)
        df["seen_in_phiusil"] = df["domain"].isin(phiusil_domains)
    else:
        df["domain"] = ""
        df["seen_in_phiusil"] = False

    seen = int(df["seen_in_phiusil"].sum())
    print(
        f"\nDomain overlap with PHIUSIL training data: "
        f"{seen:,} / {len(df):,} ({seen / max(len(df), 1) * 100:.2f}%)"
    )
    print(
        f"Genuinely unseen domains: {len(df) - seen:,} rows "
        f"({(len(df) - seen) / max(len(df), 1) * 100:.2f}%)"
    )

    for label, name in ((0, "legitimate"), (1, "phishing")):
        sub = df[df["label"] == label]
        unseen = int((~sub["seen_in_phiusil"]).sum())
        print(
            f"  {name:11s} total={len(sub):>7,}  "
            f"unseen-domain={unseen:>7,} ({unseen / max(len(sub), 1) * 100:5.1f}%)"
        )

    out = df[["url", "label", "domain", "seen_in_phiusil"]]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved external evaluation set -> {OUT_CSV}")

    stats = {
        "dataset": "LegitPhish",
        "role": "external evaluation only (never used for training)",
        "source": LEGITPHISH_LABEL_CONVENTION_SOURCE,
        "random_seed": RANDOM_SEED,
        "raw_label_convention": {
            "meaning": {str(k): v for k, v in LEGITPHISH_RAW_LABEL_MEANING.items()},
            "expected_raw_counts": {
                str(k): v for k, v in LEGITPHISH_EXPECTED_RAW_COUNTS.items()
            },
            "note": (
                "LegitPhish raw labels are 1 = legitimate, 0 = phishing, the same "
                "inverted convention as PHIUSIL. Mapped to project convention "
                "0 = legitimate, 1 = phishing."
            ),
        },
        "rows": {
            "after_dedup": int(len(df)),
            "legitimate": int(norm_counts.get(0, 0)),
            "phishing": int(norm_counts.get(1, 0)),
            "unlabelled_dropped": missing_labels,
        },
        "phiusil_overlap": {
            "pools_considered": pools_used,
            "phiusil_domains": len(phiusil_domains),
            "rows_with_domain_seen_in_phiusil": seen,
            "rows_with_unseen_domain": int(len(df) - seen),
            "pct_unseen_domain": round((len(df) - seen) / max(len(df), 1) * 100, 2),
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats written to: {OUT_STATS}")


# ═════════════════════════════════════════════════════════════════════════════
# URL-Phish (Dataset E) — the untouched FINAL external set
# ═════════════════════════════════════════════════════════════════════════════
# URL-Phish is independent of both PHIUSIL and LegitPhish:
#   * benign URLs come from the Research Organization Registry (universities and
#     research organisations), not from Wikipedia/Stack Overflow;
#   * phishing URLs come from PhishTank, not from URLHaus.
# 89% of its URLs carry a path, and its HTTPS rate is INVERTED relative to
# PHIUSIL (benign 35% HTTPS, phishing 91%), so neither the path shortcut nor the
# HTTPS shortcut can pass it.
#
# Its label convention is the INTUITIVE one (0 = benign, 1 = phishing) — the
# opposite of PHIUSIL and LegitPhish — and is asserted against the dataset card.
#
# This set was downloaded and its composition asserted before any model change,
# then left untouched. It is evaluated exactly once, at the end.

URLPHISH_RAW = pathlib.Path("data/raw/URLPhish_urls.csv")
URLPHISH_OUT_CSV = PROCESSED / "urlphish_external.csv"
URLPHISH_OUT_STATS = REPORTS / "urlphish_external_stats.json"

URLPHISH_LABEL_CONVENTION_SOURCE = "https://data.mendeley.com/datasets/65z9twcx3r/2"
URLPHISH_EXPECTED_LABEL_COUNTS = {0: 100_000, 1: 16_600}  # 0 = benign, 1 = phishing
URLPHISH_LABEL_MAP = {0: 0, 1: 1}  # already the project convention


def _url_set(path: pathlib.Path) -> set[str]:
    """Every URL string in a processed split, or an empty set if absent."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["url"].astype(str).str.strip())


def verify_urlphish_composition(counts: pd.Series) -> None:
    """
    Fail loudly unless URL-Phish's labels match its dataset card.

    The card states 100,000 benign and 16,600 phishing. Getting this backwards
    would invert the final external evaluation, so it is a hard error.
    """
    observed = {int(k): int(v) for k, v in counts.items()}
    mismatch = {
        label: (observed.get(label), expected)
        for label, expected in URLPHISH_EXPECTED_LABEL_COUNTS.items()
        if observed.get(label) != expected
    }
    if mismatch:
        detail = ", ".join(
            f"label {lab}: expected {exp:,}, found "
            f"{got if got is None else format(got, ',')}"
            for lab, (got, exp) in mismatch.items()
        )
        raise ValueError(
            f"URL-Phish label distribution does not match the dataset card ({detail}). "
            "The card states 100,000 benign (0) and 16,600 phishing (1). Verify at "
            f"{URLPHISH_LABEL_CONVENTION_SOURCE} before trusting any metric."
        )
    print(
        "  [OK] Composition verified -> 0 = benign "
        f"({observed[0]:,}), 1 = phishing ({observed[1]:,})"
    )


def prepare_urlphish() -> None:
    print()
    print("=" * 60)
    print("Preparing URL-Phish as the FINAL untouched EXTERNAL set")
    print("=" * 60)

    if not URLPHISH_RAW.exists():
        print(f"[ERROR] URL-Phish raw file not found at: {URLPHISH_RAW}")
        print("Run: python src/fraud_phishing/data/download_data.py")
        sys.exit(1)

    df = pd.read_csv(URLPHISH_RAW)
    print(f"Loaded {URLPHISH_RAW.name}: {len(df):,} rows, cols={list(df.columns)}")
    if not {"url", "label"}.issubset(df.columns):
        raise ValueError(f"Expected 'url' and 'label' columns, found {list(df.columns)}")

    df = df.dropna(subset=["url", "label"]).copy()
    counts = df["label"].astype(int).value_counts().sort_index()
    print(f"Raw label distribution:\n{counts.to_string()}")
    verify_urlphish_composition(counts)

    df["label"] = df["label"].astype(int).map(URLPHISH_LABEL_MAP)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    print(f"Dropped exact duplicate URLs: {before - len(df):,}")

    norm = df["label"].value_counts().sort_index()
    print("\nNormalized label counts:")
    print(f"  0 (benign):   {norm.get(0, 0):,}")
    print(f"  1 (phishing): {norm.get(1, 0):,}")

    # ── Overlap with everything used during development ────────────────────
    phiusil = set()
    for part in PROCESSED.glob("urls_*.csv"):
        if part.name.startswith("urls_aug") or part.name.startswith("urlphish"):
            continue
        phiusil |= _url_set(part)
    murl = _url_set(pathlib.Path("data/raw/MURL_urls.csv"))
    legitphish = _url_set(OUT_CSV)

    print(f"\nExact-URL overlap with PHIUSIL splits:   "
          f"{int(df['url'].isin(phiusil).sum()):,}")
    print(f"Exact-URL overlap with MURL (augmentation): {int(df['url'].isin(murl).sum()):,}")
    print(f"Exact-URL overlap with LegitPhish (dev):   "
          f"{int(df['url'].isin(legitphish).sum()):,}")

    df["seen_in_phiusil"] = df["url"].isin(phiusil)
    df["seen_in_murl"] = df["url"].isin(murl)
    df["seen_in_legitphish"] = df["url"].isin(legitphish)
    df["byte_identical_to_training"] = df["seen_in_phiusil"] | df["seen_in_murl"]

    import tldextract  # noqa: PLC0415

    def _domain(url: str) -> str:
        return tldextract.extract(url).top_domain_under_public_suffix or ""

    df["domain"] = df["url"].map(_domain)

    phiusil_domains, _ = load_phiusil_domains()
    df["seen_in_phiusil_domain"] = df["domain"].isin(phiusil_domains)

    for label, name in ((0, "benign"), (1, "phishing")):
        sub = df[df["label"] == label]
        mem = int(sub["byte_identical_to_training"].sum())
        unseen_dom = int((~sub["seen_in_phiusil_domain"]).sum())
        print(f"  {name:9s} total={len(sub):>7,}  "
              f"byte-identical-to-training={mem:>6,}  "
              f"unseen-PHIUSIL-domain={unseen_dom:>7,}")

    out = df[["url", "label", "domain", "seen_in_phiusil", "seen_in_murl",
              "seen_in_legitphish", "byte_identical_to_training",
              "seen_in_phiusil_domain"]]
    out.to_csv(URLPHISH_OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved final external evaluation set -> {URLPHISH_OUT_CSV}")

    stats = {
        "dataset": "URL-Phish",
        "role": "FINAL external evaluation only (untouched during development)",
        "source": URLPHISH_LABEL_CONVENTION_SOURCE,
        "label_convention": {
            "meaning": {"0": "benign", "1": "phishing"},
            "note": (
                "URL-Phish uses the intuitive convention (0 = benign, 1 = phishing), "
                "the opposite of PHIUSIL and LegitPhish. Verified against the card."
            ),
            "expected_counts": {
                str(k): v for k, v in URLPHISH_EXPECTED_LABEL_COUNTS.items()
            },
        },
        "rows": {
            "after_dedup": int(len(df)),
            "benign": int(norm.get(0, 0)),
            "phishing": int(norm.get(1, 0)),
        },
        "overlap": {
            "byte_identical_to_phiusil": int(df["seen_in_phiusil"].sum()),
            "byte_identical_to_murl": int(df["seen_in_murl"].sum()),
            "byte_identical_to_legitphish_dev": int(df["seen_in_legitphish"].sum()),
            "byte_identical_to_any_training": int(
                df["byte_identical_to_training"].sum()
            ),
            "rows_on_phiusil_training_domain": int(df["seen_in_phiusil_domain"].sum()),
        },
    }
    URLPHISH_OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats written to: {URLPHISH_OUT_STATS}")


if __name__ == "__main__":
    prepare()
    prepare_urlphish()
