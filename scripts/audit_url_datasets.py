"""
audit_url_datasets.py
=====================
P6 Step 1 + Step 3: audit every URL corpus in the project and quantify

  * how legitimate and phishing URLs differ structurally, per source and class
    (scheme, bare-host vs path, query, fragment, subdomain, length, digits,
    hyphens, suspicious tokens, brand/impersonation features);
  * duplicates, registered-domain overlap and URL-template overlap between
    sources, plus cross-source *label conflicts* (the same URL or the same
    registered domain labelled both legitimate and phishing);
  * the leakage that still exists inside the current project splits.

Read-only with respect to the model: it trains nothing, scores nothing, and
never writes to ``data/processed``. Everything lands in ``reports/p6/``.

Structural metrics are calculated from the raw string. This deliberately keeps
a missing scheme distinct from ``http`` and retains fragments, two source-format
properties that the model extractor normalises away. Registered-domain grouping
uses an offline public-suffix snapshot and never contacts a URL.

MURL is audited on a 60,000-per-class random sample (it is 661,854 rows and is
augmentation, never evaluation); overlap analysis still uses the full file.

Usage (from repo root):
    python scripts/audit_url_datasets.py
"""

import json
import pathlib
import re
import sys
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.data.prepare_urls import url_template  # noqa: E402
from fraud_phishing.features import url_features  # noqa: E402
from fraud_phishing.features.url_features import SUSPICIOUS_TOKENS  # noqa: E402

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
OUT_DIR = pathlib.Path("reports/p6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# The extractor includes bounded edit-distance checks.  A deterministic
# per-source/class sample keeps P6 practical while the separate overlap audit
# below still uses every deduplicated row.
STRUCTURAL_SAMPLE_PER_CLASS = 10_000

# ── Source registry ──────────────────────────────────────────────────────────
# label maps convert each file's raw convention to the project convention
# (0 = legitimate, 1 = phishing).
SOURCES: dict[str, dict] = {
    "PHIUSIL": {
        "path": RAW / "PHIUSIL_urls.csv",
        "url_col": "url",
        "label_col": "label_raw",
        "label_map": {1: 0, 0: 1, -1: 0},  # raw 1 = legitimate, 0 = phishing
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset",
        "collector": "Hannousse (2023)",
        "expect": {0: 134_850, 1: 100_945},
    },
    "LegitPhish": {
        "path": RAW / "LegitPhish_urls.csv",
        "url_col": "URL",
        "label_col": "ClassLabel",
        "label_map": {1.0: 0, 0.0: 1},  # raw 1 = legitimate, 0 = phishing
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/hx4m73v2sf/2",
        "collector": "Potpelwar et al. (2025)",
        "expect": {0: 37_540, 1: 63_678},
    },
    "MURL": {
        "path": RAW / "MURL_urls.csv",
        "url_col": "url",
        "label_col": "label",
        "label_map": {0: 0, 1: 1},
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/5fk24y4xmg/2",
        "collector": "Choudhary et al. (2023)",
        "expect": None,
    },
    "URL-Phish": {
        "path": RAW / "URLPhish_urls.csv",
        "url_col": "url",
        "label_col": "label",
        "label_map": {0: 0, 1: 1},
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/65z9twcx3r/2",
        "collector": "Choudhary et al. (2024)",
        "expect": {0: 100_000, 1: 16_600},
    },
}

SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://")
KEEP_FEATURES = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "digit_ratio",
    "percent_count",
    "at_count",
    "subdomain_count",
    "path_depth",
    "suspicious_token_count",
    "brand_token_count",
    "is_https",
    "is_ip_host",
    "has_query",
    "has_at_symbol",
    "has_percent_encoding",
    "is_url_shortener",
    "brand_domain_mismatch",
    "brand_lookalike_domain",
)

_extract_cache: dict[str, str] = {}
# Use tldextract's packaged suffix snapshot only.  This makes the audit
# reproducible offline and prevents the audit itself from contacting a URL or
# attempting a suffix-list refresh.
OFFLINE_TLD_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=str(OUT_DIR / ".tldextract")
)
url_features.tldextract.extract = OFFLINE_TLD_EXTRACT


def _registered_domain(url: str) -> str:
    try:
        parsed = urlsplit(url if raw_scheme(url) != "__none__" else f"//{url}")
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    if not host:
        return "__none__"
    if host not in _extract_cache:
        try:
            ext = OFFLINE_TLD_EXTRACT(host)
            _extract_cache[host] = ext.top_domain_under_public_suffix or "__none__"
        except Exception:
            _extract_cache[host] = "__none__"
    return _extract_cache[host]


def log(msg: str, log_lines: list[str]) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def raw_scheme(url: str) -> str:
    match = SCHEME_RE.match(url.strip())
    return match.group(1).lower() if match else "__none__"


def load_source(name: str, spec: dict, log_lines: list[str]) -> pd.DataFrame:
    path = spec["path"]
    if not path.exists():
        log(f"[WARN] {name}: {path} not found - skipped", log_lines)
        return pd.DataFrame(columns=["url", "label"])
    df = pd.read_csv(path, low_memory=False)
    df = df[[spec["url_col"], spec["label_col"]]].rename(
        columns={spec["url_col"]: "url", spec["label_col"]: "label_raw"}
    )
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].ne("") & df["url"].str.lower().ne("nan")]
    df = df.dropna(subset=["label_raw"])

    def _map(v):
        for key, mapped in spec["label_map"].items():
            if v == key or str(v) == str(key):
                return mapped
        return None

    df["label"] = df["label_raw"].map(_map)
    unknown = int(df["label"].isna().sum())
    if unknown:
        log(f"[WARN] {name}: dropping {unknown:,} rows with unmapped labels", log_lines)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    log(
        f"{name:<12} rows={len(df):>7,}  (dropped {before - len(df):,} exact dup rows)"
        f"  legit={int((df['label'] == 0).sum()):>7,}"
        f"  phishing={int((df['label'] == 1).sum()):>7,}",
        log_lines,
    )
    if spec["expect"]:
        got = {int(k): int((df["label"] == int(k)).sum()) for k in (0, 1)}
        if got != spec["expect"]:
            log(
                f"  [WARN] composition {got} != documented {spec['expect']} "
                "(check the label convention before trusting any metric)",
                log_lines,
            )
        else:
            log(f"  [OK] composition matches the documented card {got}", log_lines)
    return df[["url", "label"]].reset_index(drop=True)


# ── Structural audit ─────────────────────────────────────────────────────────

def _raw_metrics(url: str) -> dict[str, float]:
    """Cheap raw-string structural metrics; does not fetch a URL."""
    raw = url.strip()
    scheme = raw_scheme(raw)
    try:
        parsed = urlsplit(raw if scheme != "__none__" else f"http://{raw}")
    except ValueError:
        parsed = urlsplit("")
    host = (parsed.hostname or "").rstrip(".")
    labels = [part for part in host.split(".") if part]
    lower = raw.lower()
    return {
        "with_path": float(bool(parsed.path.strip("/"))),
        "has_query": float(bool(parsed.query)),
        "path_len": float(len(parsed.path)),
        "subdomain_proxy": float(len(labels) > 2),
        "url_len": float(len(raw)),
        "digits": float(sum(c.isdigit() for c in raw)),
        "hyphens": float(raw.count("-")),
        "suspicious": float(sum(token in lower for token in SUSPICIOUS_TOKENS)),
    }


def structural_audit(
    frame: pd.DataFrame, name: str, log_lines: list[str]
) -> pd.DataFrame:
    """Per-class structural summary for one source."""
    t0 = time.perf_counter()
    schemes = frame["url"].map(raw_scheme)
    has_fragment = frame["url"].str.contains("#", regex=False, na=False)
    rows = []
    for label in (0, 1):
        sub = frame[frame["label"] == label]
        if sub.empty:
            continue
        feats = pd.DataFrame([_raw_metrics(u) for u in sub["url"].tolist()])
        schemes_sub = schemes[sub.index]
        frag = has_fragment[sub.index].to_numpy()
        n = len(feats)
        with_path = feats["with_path"].to_numpy(dtype=bool)
        record = {
            "source": name,
            "class": "legitimate" if label == 0 else "phishing",
            "n": n,
            "pct_no_scheme": round(float((schemes_sub == "__none__").mean()) * 100, 2),
            "pct_https": round(float((schemes_sub == "https").mean()) * 100, 2),
            "pct_http": round(float((schemes_sub == "http").mean()) * 100, 2),
            "pct_other_scheme": round(
                float(
                    (~schemes_sub.isin(["https", "http", "__none__"])).mean()
                )
                * 100,
                2,
            ),
            "pct_with_path": round(float(with_path.mean()) * 100, 2),
            "pct_query": round(float(feats["has_query"].mean()) * 100, 2),
            "pct_fragment": round(float(frag.mean()) * 100, 2),
            "pct_subdomain_proxy": round(float(feats["subdomain_proxy"].mean()) * 100, 2),
            "pct_suspicious_token": round(
                float((feats["suspicious"] > 0).mean()) * 100, 2
            ),
            "url_len_mean": round(float(feats["url_len"].mean()), 1),
            "url_len_median": float(feats["url_len"].median()),
            "url_len_p90": float(feats["url_len"].quantile(0.90)),
            "path_len_mean": round(float(feats["path_len"].mean()), 1),
            "digits_mean": round(float(feats["digits"].mean()), 2),
            "hyphens_mean": round(float(feats["hyphens"].mean()), 2),
            "suspicious_tokens_mean": round(
                float(feats["suspicious"].mean()), 2
            ),
        }
        rows.append(record)
        log(
            f"  {name}/{record['class']:<10} n={n:>7,}  "
            f"https={record['pct_https']:>5.1f}%  no_scheme={record['pct_no_scheme']:>5.1f}%"
            f"  path={record['pct_with_path']:>5.1f}%  query={record['pct_query']:>5.1f}%"
            f"  subdom={record['pct_subdomain_proxy']:>5.1f}%  len_med={record['url_len_median']:.0f}",
            log_lines,
        )
    log(f"  ({name} structural audit took {time.perf_counter() - t0:.1f}s)", log_lines)
    return pd.DataFrame(rows)


def scheme_path_crosstab(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """The conjunction table that exposes the dataset shortcut."""
    scheme = frame["url"].map(raw_scheme).replace({"__none__": "no-scheme"})
    has_path = frame["url"].map(lambda u: bool(_raw_metrics(u)["with_path"]))
    tmp = pd.DataFrame({"scheme": scheme, "has_path": has_path, "label": frame["label"]})
    tmp["shape"] = np.where(tmp["has_path"], "with path", "bare host")
    out = (
        tmp.groupby(["label", "scheme", "shape"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    out["source"] = name
    out["class"] = out["label"].map({0: "legitimate", 1: "phishing"})
    return out[["source", "class", "scheme", "shape", "n"]]


# ── Overlap audit ────────────────────────────────────────────────────────────

def overlap_audit(frames: dict[str, pd.DataFrame], log_lines: list[str]) -> dict:
    urls = {name: set(df["url"]) for name, df in frames.items() if len(df)}
    labels: dict[str, dict[str, int]] = {}
    domains: dict[str, dict[str, set[str]]] = {}
    templates: dict[str, set[str]] = {}

    for name, df in frames.items():
        if not len(df):
            continue
        labels[name] = dict(zip(df["url"], df["label"]))
        domain_map: dict[str, set[str]] = {}
        for url, label in zip(df["url"], df["label"]):
            dom = _registered_domain(url)
            domain_map.setdefault(dom, set()).add(str(label))
        domains[name] = domain_map
        templates[name] = {url_template(u) for u in df["url"]}
        log(f"  {name:<12} unique urls={len(urls[name]):>7,}  "
            f"domains={len(domain_map):>7,}  templates={len(templates[name]):>7,}",
            log_lines)

    pairs = []
    names = [n for n in frames if len(frames[n])]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = urls[a] & urls[b]
            conflicts = [
                u for u in common if labels[a][u] != labels[b][u]
            ]
            dom_a, dom_b = domains[a], domains[b]
            shared_domains = set(dom_a) & set(dom_b)
            domain_label_conflicts = [
                d
                for d in shared_domains
                if dom_a[d] != dom_b[d] and "__none__" not in d
            ]
            tmpl_common = templates[a] & templates[b]
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "exact_url_overlap": len(common),
                    "exact_url_overlap_pct_of_b": round(len(common) / max(len(urls[b]), 1) * 100, 3),
                    "exact_url_label_conflicts": len(conflicts),
                    "exact_url_conflict_examples": conflicts[:5],
                    "shared_registered_domains": len(shared_domains - {"__none__"}),
                    "shared_domain_label_conflicts": len(domain_label_conflicts),
                    "shared_domain_conflict_examples": sorted(domain_label_conflicts)[:5],
                    "shared_templates": len(tmpl_common),
                }
            )
            log(
                f"  {a:<12} vs {b:<12} exact={len(common):>7,} "
                f"label_conflicts={len(conflicts):>6,} "
                f"shared_domains={len(shared_domains - {'__none__'}):>7,} "
                f"domain_conflicts={len(domain_label_conflicts):>6,} "
                f"shared_templates={len(tmpl_common):>7,}",
                log_lines,
            )

    # within-source domain label conflicts (same domain, both classes)
    within = {}
    for name, domain_map in domains.items():
        within[name] = {
            "domains_with_both_labels": sum(
                1 for d, lab in domain_map.items() if len(lab) > 1 and d != "__none__"
            ),
            "domains_total": len(domain_map),
        }
    return {"pairs": pairs, "within_source": within}


def split_integrity(log_lines: list[str]) -> dict:
    """Recompute exact/domain/template overlap for the current project splits."""
    split_files = {
        "urls_train.csv": "train (random)",
        "urls_val.csv": "val (random)",
        "urls_test.csv": "test (random)",
        "urls_domain_train.csv": "domain train",
        "urls_domain_test.csv": "domain test",
        "urls_template_test.csv": "template-clean test",
        "urls_aug_train.csv": "augmented fit",
        "urls_aug_val.csv": "augmented val",
        "legitphish_external.csv": "LegitPhish (dev)",
        "urlphish_external.csv": "URL-Phish (dev-informed)",
    }
    data = {}
    for filename, label in split_files.items():
        path = PROCESSED / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        data[label] = {
            "file": filename,
            "n": int(len(df)),
            "legit": int((df["label"] == 0).sum()),
            "phish": int((df["label"] == 1).sum()),
            "urls": set(df["url"].astype(str)),
            "domains": {_registered_domain(u) for u in df["url"].astype(str)} - {"__none__"},
            "templates": {url_template(u) for u in df["url"].astype(str)},
        }
        log(f"  loaded {filename:<28} n={len(df):>7,}", log_lines)

    checks = [
        ("train (random)", "test (random)"),
        ("train (random)", "val (random)"),
        ("val (random)", "test (random)"),
        ("domain train", "domain test"),
        ("domain train", "template-clean test"),
        ("augmented fit", "domain test"),
        ("augmented fit", "template-clean test"),
        ("augmented fit", "test (random)"),
        ("augmented fit", "LegitPhish (dev)"),
        ("augmented fit", "URL-Phish (dev-informed)"),
        ("domain train", "URL-Phish (dev-informed)"),
        ("domain train", "LegitPhish (dev)"),
    ]
    reports = []
    for a, b in checks:
        if a not in data or b not in data:
            continue
        exact = len(data[a]["urls"] & data[b]["urls"])
        shared_dom = data[a]["domains"] & data[b]["domains"]
        tmpl = len(data[a]["templates"] & data[b]["templates"])
        b_domains = data[b]["domains"]
        reports.append(
            {
                "a": a,
                "b": b,
                "expectation": EXPECTATIONS.get((a, b), "informational"),
                "exact_url_overlap": exact,
                "shared_registered_domains": len(shared_dom),
                "b_domains": len(b_domains),
                "b_pct_domains_seen_in_a": round(
                    len(shared_dom) / max(len(b_domains), 1) * 100, 2
                ),
                "shared_templates": tmpl,
                "b_pct_templates_seen_in_a": round(
                    tmpl / max(len(data[b]["templates"]), 1) * 100, 2
                ),
            }
        )
        log(
            f"  {a:<26} -> {b:<26} exact={exact:>7,} "
            f"shared_domains={len(shared_dom):>7,} "
            f"({reports[-1]['b_pct_domains_seen_in_a']:>5.1f}% of b) "
            f"templates={tmpl:>7,}",
            log_lines,
        )
    return {
        "splits": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("urls", "domains", "templates")}
            for k, v in data.items()
        },
        "checks": reports,
    }


EXPECTATIONS = {
    ("train (random)", "test (random)"): "must be 0 exact (random split is disjoint by URL)",
    ("train (random)", "val (random)"): "must be 0 exact",
    ("val (random)", "test (random)"): "must be 0 exact",
    ("domain train", "domain test"): "must be 0 exact and 0 shared domains (hard requirement)",
    ("domain train", "template-clean test"): "must be 0 exact, domains and templates",
    ("augmented fit", "domain test"): "must be disjoint by domain",
    ("augmented fit", "template-clean test"): "must be disjoint by domain and template",
    ("augmented fit", "URL-Phish (dev-informed)"): "informational - 595 known overlaps",
}


def main() -> None:
    log_lines: list[str] = []
    log(f"P6 URL dataset audit  ({pd.Timestamp.now():%Y-%m-%d %H:%M:%S})", log_lines)
    log("=" * 78, log_lines)

    # ── Load ─────────────────────────────────────────────────────────────────
    log("\n[1] Loading sources", log_lines)
    frames: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict] = {}
    for name, spec in SOURCES.items():
        df = load_source(name, spec, log_lines)
        if not len(df):
            continue
        frames[name] = df
        summary[name] = {
            "file": str(spec["path"]),
            "rows_after_dedup": int(len(df)),
            "legitimate": int((df["label"] == 0).sum()),
            "phishing": int((df["label"] == 1).sum()),
            "licence": spec["license"],
            "source_page": spec["source"],
            "collector": spec["collector"],
        }

    # ── Structural audit (deterministic per-class samples) ───────────────────
    log("\n[2] Structural audit (raw URL strings)", log_lines)
    struct_parts = []
    crosstab_parts = []
    for name, df in frames.items():
        audit_df = df
        rng = np.random.default_rng(RANDOM_SEED)
        parts = []
        for label in (0, 1):
            sub = df[df["label"] == label]
            if len(sub) > STRUCTURAL_SAMPLE_PER_CLASS:
                idx = rng.choice(len(sub), STRUCTURAL_SAMPLE_PER_CLASS, replace=False)
                sub = sub.iloc[np.sort(idx)]
            parts.append(sub)
        audit_df = pd.concat(parts)
        if len(audit_df) != len(df):
            log(
                f"  NOTE {name} is audited on a {STRUCTURAL_SAMPLE_PER_CLASS:,}-per-class "
                f"random sample ({len(audit_df):,} rows); overlaps use the full file",
                log_lines,
            )
        struct_parts.append(structural_audit(audit_df, name, log_lines))
        crosstab_parts.append(scheme_path_crosstab(audit_df, name))

    struct = pd.concat(struct_parts, ignore_index=True)
    struct.to_csv(OUT_DIR / "structural_by_source_class.csv", index=False)
    crosstab = pd.concat(crosstab_parts, ignore_index=True)
    crosstab.to_csv(OUT_DIR / "scheme_path_crosstab.csv", index=False)

    # ── Overlap audit ────────────────────────────────────────────────────────
    log("\n[3] Duplicate / domain / template overlap between sources", log_lines)
    overlaps = overlap_audit(frames, log_lines)

    # ── Split integrity ──────────────────────────────────────────────────────
    log("\n[4] Current project split integrity", log_lines)
    integrity = split_integrity(log_lines)

    # ── Write ────────────────────────────────────────────────────────────────
    (OUT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "overlaps.json").write_text(
        json.dumps(overlaps, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "split_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "audit_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nWrote outputs to {OUT_DIR}", log_lines)


if __name__ == "__main__":
    main()
