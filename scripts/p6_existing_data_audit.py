"""Fast, read-only P6 structural audit of the project URL corpora.

Writes per-class descriptive statistics to reports/p6/structural_distributions.csv.
It reads only URL and label columns and does not train, score, or alter models.
"""
from __future__ import annotations

import pathlib
import re
import sys
from urllib.parse import urlsplit

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.features.url_features import SUSPICIOUS_TOKENS  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "p6" / "structural_distributions.csv"
SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://")

SOURCES = {
    "PHIUSIL": ("PHIUSIL_urls.csv", "url", "label_raw", {1: 0, 0: 1}),
    "LegitPhish": ("LegitPhish_urls.csv", "URL", "ClassLabel", {1: 0, 0: 1}),
    "URL-Phish": ("URLPhish_urls.csv", "url", "label", {0: 0, 1: 1}),
    "MURL": ("MURL_urls.csv", "url", "label", {0: 0, 1: 1}),
}


def parse_url(raw: str) -> dict:
    raw = raw.strip()
    scheme = (SCHEME.match(raw).group(1).lower() if SCHEME.match(raw) else "none")
    try:
        parsed = urlsplit(raw if scheme != "none" else "http://" + raw)
    except ValueError:  # malformed bracketed IPv6 / malformed authority
        parsed = urlsplit("")
    host = (parsed.hostname or "").lower().rstrip(".")
    labels = [part for part in host.split(".") if part]
    lower = raw.lower()
    return {
        "scheme": scheme,
        "with_path": int(bool(parsed.path.strip("/"))),
        "query": int(bool(parsed.query)),
        "fragment": int(bool(parsed.fragment)),
        # A syntactic proxy only; it does not use the public suffix list.
        "subdomain_proxy": int(len(labels) > 2),
        "url_length": len(raw),
        "digits": sum(ch.isdigit() for ch in raw),
        "hyphens": raw.count("-"),
        "suspicious_tokens": sum(token in lower for token in SUSPICIOUS_TOKENS),
    }


def main() -> None:
    results = []
    for name, (filename, url_col, label_col, label_map) in SOURCES.items():
        path = RAW / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[[url_col, label_col]].dropna().copy()
        frame[url_col] = frame[url_col].astype(str).str.strip()
        frame = frame[frame[url_col].ne("") & frame[url_col].str.lower().ne("nan")]
        frame["label"] = frame[label_col].map(label_map)
        frame = frame.dropna(subset=["label"])
        frame["label"] = frame["label"].astype(int)
        frame = frame.drop_duplicates(subset=[url_col])

        parsed = pd.DataFrame.from_records(frame[url_col].map(parse_url))
        parsed["label"] = frame["label"].to_numpy()
        for label, group in parsed.groupby("label"):
            n = len(group)
            results.append({
                "source": name,
                "class": "legitimate" if label == 0 else "phishing",
                "n": n,
                "pct_http": round(100 * (group.scheme == "http").mean(), 2),
                "pct_https": round(100 * (group.scheme == "https").mean(), 2),
                "pct_no_scheme": round(100 * (group.scheme == "none").mean(), 2),
                "pct_with_path": round(100 * group.with_path.mean(), 2),
                "pct_query": round(100 * group["query"].mean(), 2),
                "pct_fragment": round(100 * group.fragment.mean(), 2),
                "pct_subdomain_proxy": round(100 * group.subdomain_proxy.mean(), 2),
                "url_length_mean": round(group.url_length.mean(), 2),
                "url_length_median": round(group.url_length.median(), 2),
                "digits_mean": round(group.digits.mean(), 3),
                "hyphens_mean": round(group.hyphens.mean(), 3),
                "suspicious_tokens_mean": round(group.suspicious_tokens.mean(), 3),
            })
        print(f"{name}: {len(frame):,} distinct non-empty labeled URLs", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
