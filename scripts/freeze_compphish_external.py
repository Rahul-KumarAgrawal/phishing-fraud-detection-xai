"""Create a deterministic, leakage-controlled CompPhish external candidate.

Static URL data only. This script never visits URLs and never loads a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import sys
from datetime import datetime, timezone

import pandas as pd
import tldextract

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_compphish_overlap import hostname, normalized_url  # noqa: E402
from audit_url_datasets import SOURCES, _registered_domain, load_source  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.data.prepare_urls import url_template  # noqa: E402

RAW = ROOT / "data" / "raw"
MAPPING = RAW / "Mapping_File.xlsx"
OUT = ROOT / "data" / "processed" / "compphish_external_frozen.csv"
MANIFEST = ROOT / "reports" / "p6" / "compphish_external_frozen_manifest.json"
LOG = ROOT / "reports" / "p6" / "compphish_external_frozen.log"
RULE_ORDER = ("malformed", "exact_url", "normalized_url", "hostname", "registered_domain", "template")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def development_keys() -> dict[str, set[str]]:
    result = {key: set() for key in RULE_ORDER[1:]}
    for name, spec in SOURCES.items():
        source = load_source(name, spec, [])
        for url in source["url"]:
            result["exact_url"].add(url)
            result["normalized_url"].add(normalized_url(url))
            result["hostname"].add(hostname(url))
            result["registered_domain"].add(_registered_domain(url))
            result["template"].add(url_template(url))
    return result


def main() -> None:
    if not MAPPING.exists():
        raise FileNotFoundError(MAPPING)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        log_lines.append(line)
        LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    raw = pd.read_excel(MAPPING, usecols=["serial_number", "url", "label"])
    raw["url"] = raw["url"].astype(str).str.strip()
    raw["label"] = raw["label"].astype(int)
    # Deterministic exact dedup: lowest serial number wins. A conflicting
    # duplicate would be excluded under the source_label_conflict flag.
    raw = raw.sort_values("serial_number", kind="stable").reset_index(drop=True)
    duplicate_labels = raw.groupby("url")["label"].nunique()
    conflicted_source_urls = set(duplicate_labels[duplicate_labels > 1].index)
    duplicate_rows_removed = int(raw.duplicated("url", keep="first").sum())
    unique = raw.drop_duplicates("url", keep="first").copy()
    log("building approved-development key sets")
    dev = development_keys()
    flags: list[dict[str, bool]] = []
    for url in unique["url"]:
        host = hostname(url)
        flags.append({
            "malformed": host == "__none__",
            "exact_url": url in dev["exact_url"],
            "normalized_url": normalized_url(url) in dev["normalized_url"],
            "hostname": host in dev["hostname"],
            "registered_domain": _registered_domain(url) in dev["registered_domain"],
            "template": url_template(url) in dev["template"],
        })
    flag_frame = pd.DataFrame(flags, index=unique.index)
    unique = pd.concat([unique, flag_frame], axis=1)
    unique["source_label_conflict"] = unique["url"].isin(conflicted_source_urls)
    primary = []
    for _, row in unique.iterrows():
        if row["source_label_conflict"]:
            primary.append("source_label_conflict")
            continue
        primary.append(next((rule for rule in RULE_ORDER if bool(row[rule])), "retained"))
    unique["primary_disposition"] = primary
    retained = unique[unique["primary_disposition"] == "retained"][["serial_number", "url", "label"]]
    retained = retained.sort_values("serial_number", kind="stable")
    csv_bytes = retained.to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_bytes(OUT, csv_bytes)
    post = {rule: 0 for rule in RULE_ORDER[1:]}
    for url in retained["url"]:
        post["exact_url"] += int(url in dev["exact_url"])
        post["normalized_url"] += int(normalized_url(url) in dev["normalized_url"])
        post["hostname"] += int(hostname(url) in dev["hostname"])
        post["registered_domain"] += int(_registered_domain(url) in dev["registered_domain"])
        post["template"] += int(url_template(url) in dev["template"])
    if any(post.values()):
        raise RuntimeError(f"Frozen candidate still overlaps development corpus: {post}")
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"dataset": "CompPhish v4", "mapping_filename": MAPPING.name, "mapping_sha256": sha256(MAPPING)},
        "other_original_files": {name: sha256(RAW / name) for name in ["Readme.docx", "Data_Dictionary.xlsx", "All_Features_threshold90.xlsx"]},
        "development_boundary": list(SOURCES),
        "exposure_before_freeze": "CompPhish structural analysis was performed in P6.1; no model predictions, fitting, feature selection, or threshold work used this candidate.",
        "definitions": {
            "dedup": "sort serial_number ascending and retain first exact trimmed URL",
            "normalized_url": "lowercase scheme and host; remove fragment and default HTTP/HTTPS port; retain path/query",
            "hostname": "lowercase parsed hostname",
            "registered_domain": "offline tldextract public suffix snapshot",
            "template": "project url_template: lowercase, remove scheme/query/fragment and trailing slash, mask digit runs",
        },
        "counts": {
            "raw_rows": int(len(raw)), "exact_duplicate_rows_removed": duplicate_rows_removed,
            "unique_urls_pre_filter": int(len(unique)),
            "original_unique_class_counts": {str(k): int(v) for k, v in unique["label"].value_counts().sort_index().items()},
            "retained_rows": int(len(retained)),
            "retained_class_counts": {str(k): int(v) for k, v in retained["label"].value_counts().sort_index().items()},
            "retained_unique_hostnames": len({hostname(url) for url in retained["url"]}),
            "retained_unique_registered_domains": len({_registered_domain(url) for url in retained["url"]}) - int("__none__" in {_registered_domain(url) for url in retained["url"]}),
            "retained_unique_templates": len({url_template(url) for url in retained["url"]}),
            "flag_counts_nonexclusive": {rule: int(unique[rule].sum()) for rule in RULE_ORDER},
            "primary_disposition_counts": {str(k): int(v) for k, v in unique["primary_disposition"].value_counts().items()},
        },
        "post_filter_overlap": post,
        "output": {"filename": str(OUT.relative_to(ROOT)), "sha256": sha256(OUT)},
        "reproduce": "python scripts/freeze_compphish_external.py",
        "environment": {"python": sys.version, "pandas": pd.__version__, "tldextract": tldextract.__version__, "platform": platform.platform()},
    }
    atomic_bytes(MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    log(f"frozen {len(retained):,} rows; sha256={manifest['output']['sha256']}")


if __name__ == "__main__":
    main()
