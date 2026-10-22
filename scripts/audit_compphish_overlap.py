"""Resumable, static-only P6.1 CompPhish overlap audit.

Writes atomic progress after every source comparison. It reads raw URL strings
only and never visits destinations or uses HTML-derived feature fields.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.data.prepare_urls import url_template  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from audit_url_datasets import SOURCES, _registered_domain, load_source  # noqa: E402

OUT = ROOT / "reports" / "p6" / "compphish_overlap.json"
LOG = ROOT / "reports" / "p6" / "compphish_overlap.log"
MAPPING = ROOT / "data" / "raw" / "Mapping_File.xlsx"


def hostname(url: str) -> str:
    try:
        p = urlsplit(url if "://" in url else f"//{url}")
        return (p.hostname or "").lower().rstrip(".") or "__none__"
    except ValueError:
        return "__none__"


def normalized_url(url: str) -> str:
    """Conservative URL key: lowercase scheme/host, remove fragment/default port."""
    raw = url.strip()
    try:
        p = urlsplit(raw if "://" in raw else f"//{raw}")
        host = (p.hostname or "").lower().rstrip(".")
        scheme = p.scheme.lower()
        port = p.port
        netloc = host
        if p.username:
            netloc = f"{p.username}@{netloc}"
        if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{netloc}:{port}"
        return urlunsplit((scheme, netloc, p.path or "", p.query or "", ""))
    except ValueError:
        return raw.lower().split("#", 1)[0]


def keyed(frame: pd.DataFrame, key_fn) -> tuple[set[str], dict[str, set[int]]]:
    labels: dict[str, set[int]] = {}
    for url, label in zip(frame["url"], frame["label"]):
        key = key_fn(url)
        if key == "__none__":
            continue
        labels.setdefault(key, set()).add(int(label))
    return set(labels), labels


def atomic_write(payload: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, OUT)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    def log(message: str) -> None:
        stamp = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(stamp, flush=True)
        log_lines.append(stamp)
        LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not MAPPING.exists():
        raise FileNotFoundError(f"Missing CompPhish mapping file: {MAPPING}")
    candidate = pd.read_excel(MAPPING, usecols=["url", "label"])
    candidate["url"] = candidate["url"].astype(str).str.strip()
    candidate = candidate[candidate["url"].ne("")].copy()
    candidate["label"] = candidate["label"].astype(int)
    raw_rows = int(len(candidate))
    candidate = candidate.drop_duplicates("url").reset_index(drop=True)
    payload = {
        "status": "in_progress",
        "method": {
            "exact_raw_url": "trimmed raw URL string",
            "normalized_url": "lowercase scheme/host, fragment removed, default HTTP/HTTPS ports removed; path and query retained",
            "hostname": "lowercase parsed hostname",
            "registered_domain": "offline public-suffix extraction",
            "template": "project url_template: lowercase, scheme/query/fragment removed, trailing slash removed, digit runs masked",
        },
        "comp_phish": {"raw_rows": raw_rows, "unique_exact_urls": int(len(candidate)), "labels": candidate["label"].value_counts().sort_index().to_dict()},
        "comparisons": [],
    }
    atomic_write(payload)
    candidate_levels = {name: keyed(candidate, fn) for name, fn in {
        "exact_raw_url": lambda x: x,
        "normalized_url": normalized_url,
        "hostname": hostname,
        "registered_domain": _registered_domain,
        "template": url_template,
    }.items()}
    all_source_labels: dict[str, dict[str, set[int]]] = {
        level: {} for level in candidate_levels
    }
    for name, spec in SOURCES.items():
        log(f"loading {name}")
        source = load_source(name, spec, [])
        item = {"source": name, "raw_rows_after_source_cleaning": int(len(source)), "levels": {}}
        for level, (_, candidate_labels) in candidate_levels.items():
            fn = {"exact_raw_url": lambda x: x, "normalized_url": normalized_url, "hostname": hostname, "registered_domain": _registered_domain, "template": url_template}[level]
            source_keys, source_labels = keyed(source, fn)
            for key, labels in source_labels.items():
                all_source_labels[level].setdefault(key, set()).update(labels)
            candidate_keys = set(candidate_labels)
            shared = candidate_keys & source_keys
            conflicts = sum(1 for key in shared if candidate_labels[key] != source_labels[key])
            item["levels"][level] = {
                "comp_phish_unique": len(candidate_keys),
                "source_unique": len(source_keys),
                "intersection": len(shared),
                "pct_of_comp_phish": round(100 * len(shared) / max(len(candidate_keys), 1), 4),
                "pct_of_source": round(100 * len(shared) / max(len(source_keys), 1), 4),
                "conflicting_label_keys": conflicts,
                "conflict_examples": sorted(k for k in shared if candidate_labels[k] != source_labels[k])[:5],
            }
        payload["comparisons"].append(item)
        atomic_write(payload)
        log(f"completed {name}")
    retained = {}
    for level, (_, candidate_labels) in candidate_levels.items():
        overlaps = set(candidate_labels) & set(all_source_labels[level])
        retained[level] = {
            "candidate_keys_overlapping_any_existing_source": len(overlaps),
            "candidate_keys_retained_if_excluding_this_level": len(candidate_labels) - len(overlaps),
            "candidate_keys_with_any_label_conflict": sum(
                1 for key in overlaps if candidate_labels[key] != all_source_labels[level][key]
            ),
        }
    payload["proposed_single_level_exclusions"] = retained
    payload["status"] = "complete"
    atomic_write(payload)
    log("completed all comparisons")


if __name__ == "__main__":
    main()
