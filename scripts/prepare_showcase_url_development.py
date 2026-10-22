"""Prepare a deterministic, leakage-resistant URL corpus for the final showcase.

Inputs are PHIUSIIL, conflict-screened MURL, and a byte-identical development
copy of the previously evaluated CompPhish frozen subset.  The P9 frozen input,
its manifest, and historical P9 outputs are read only and never overwritten.

Every split boundary separates connected components induced by registered domain
or the project-normalized URL template.  Exact and normalized URL overlaps are
also asserted to be zero.  No URL is fetched, resolved, or otherwise visited.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_compphish_overlap import normalized_url  # noqa: E402
from audit_url_datasets import SOURCES, _registered_domain, load_source  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.data.prepare_urls import url_template  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "showcase"
FROZEN = DATA / "compphish_external_frozen.csv"
FROZEN_MANIFEST = ROOT / "reports" / "p6" / "compphish_external_frozen_manifest.json"
COMP_DEV_COPY = DATA / "showcase_compphish_development.csv"
SEED = 42
MURL_CAP_PER_CLASS = 60_000
EXPECTED_FROZEN_SHA256 = "56f6e8545a7df5f675b740d0858ee0fb4b2fa920a085d4aceae2838bfbc0e377"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Write a byte-identical development copy without exposing a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temp_path = pathlib.Path(temp.name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_csv(frame: pd.DataFrame, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, suffix=".csv", delete=False, newline="", encoding="utf-8") as temp:
        temp_path = pathlib.Path(temp.name)
        frame.to_csv(temp, index=False)
    try:
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def component_groups(frame: pd.DataFrame) -> np.ndarray:
    groups = UnionFind(len(frame))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(frame[["domain", "template"]].itertuples(index=False)):
        for kind, value in (("domain", row.domain), ("template", row.template)):
            key = (kind, value)
            prior = seen.get(key)
            if prior is None:
                seen[key] = index
            else:
                groups.union(index, prior)
    return np.asarray([groups.find(index) for index in range(len(frame))])


def summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "class_counts": {str(label): int(count) for label, count in frame.label.value_counts().sort_index().items()},
        "source_counts": {str(source): int(count) for source, count in frame.source.value_counts().sort_index().items()},
        "unique_raw_urls": int(frame.url.nunique()),
        "unique_normalized_urls": int(frame.normalized.nunique()),
        "unique_registered_domains": int(frame.domain.nunique()),
        "unique_templates": int(frame.template.nunique()),
    }


def pair_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    return {
        "exact_url": int(len(set(left.url) & set(right.url))),
        "normalized_url": int(len(set(left.normalized) & set(right.normalized))),
        "registered_domain": int(len(set(left.domain) & set(right.domain))),
        "template": int(len(set(left.template) & set(right.template))),
    }


def _load_comp_phish() -> pd.DataFrame:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    frozen_hash = sha256(FROZEN)
    if frozen_hash != EXPECTED_FROZEN_SHA256 or frozen_hash != manifest["output"]["sha256"]:
        raise RuntimeError("CompPhish frozen input hash differs from the recorded P6/P9 hash")
    frame = pd.read_csv(FROZEN)
    if list(frame.columns) != ["serial_number", "url", "label"]:
        raise RuntimeError("CompPhish frozen input schema is not the recorded P6 schema")
    if not set(frame.label.unique()).issubset({0, 1}) or frame.url.isna().any():
        raise RuntimeError("CompPhish frozen input has invalid labels or URLs")
    atomic_copy(FROZEN, COMP_DEV_COPY)
    if sha256(COMP_DEV_COPY) != frozen_hash:
        raise RuntimeError("CompPhish development copy is not byte-identical to frozen input")
    return frame[["url", "label"]].assign(source="CompPhish")


def _screen_murl(murl: pd.DataFrame, comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quarantine MURL exact-label contradictions against all known diagnostics."""
    references: list[pd.DataFrame] = [comp[["url", "label"]]]
    for name in ("LegitPhish", "URL-Phish"):
        references.append(load_source(name, SOURCES[name], []))
    reference_labels: dict[str, set[int]] = {}
    for reference in references:
        for url, label in reference[["url", "label"]].itertuples(index=False):
            reference_labels.setdefault(str(url), set()).add(int(label))
    conflict_urls = {
        str(url)
        for url, label in murl[["url", "label"]].itertuples(index=False)
        if int(label) not in reference_labels.get(str(url), {int(label)})
    }
    screened = murl[~murl.url.isin(conflict_urls)].copy()
    return screened, {"murl_exact_label_conflicts_quarantined": int(len(conflict_urls))}


def _cap_murl(murl: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pieces: list[pd.DataFrame] = []
    for label in (0, 1):
        part = murl[murl.label == label].sort_values("url", kind="stable")
        if len(part) > MURL_CAP_PER_CLASS:
            selected = np.sort(rng.choice(len(part), MURL_CAP_PER_CLASS, replace=False))
            part = part.iloc[selected]
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    phiusil = load_source("PHIUSIL", SOURCES["PHIUSIL"], []).assign(source="PHIUSIL")
    murl = load_source("MURL", SOURCES["MURL"], []).assign(source="MURL")
    comp = _load_comp_phish()
    murl, conflict_info = _screen_murl(murl, comp)
    murl_before_cap = len(murl)
    murl = _cap_murl(murl)

    combined = pd.concat([comp, phiusil, murl], ignore_index=True)
    combined["url"] = combined.url.astype(str).str.strip()
    combined = combined[combined.url.ne("") & combined.url.str.lower().ne("nan")].copy()
    combined["normalized"] = combined.url.map(normalized_url)

    # A normalized URL with contradictory labels is removed from every source.
    norm_label_count = combined.groupby("normalized").label.nunique()
    conflicting_normalized = set(norm_label_count[norm_label_count > 1].index)
    before_norm_conflicts = len(combined)
    combined = combined[~combined.normalized.isin(conflicting_normalized)].copy()

    # For consistent-label exact/normalized duplicates, retain one deterministic
    # row. CompPhish gets precedence so its development contribution is explicit.
    source_order = {"CompPhish": 0, "PHIUSIL": 1, "MURL": 2}
    combined["source_order"] = combined.source.map(source_order)
    combined = combined.sort_values(["normalized", "source_order", "url"], kind="stable")
    before_dedup = len(combined)
    combined = combined.drop_duplicates("normalized", keep="first").copy()
    combined["domain"] = combined.url.map(_registered_domain)
    combined["template"] = combined.url.map(url_template)
    malformed = combined[(combined.domain == "__none__") | combined.template.eq("")]
    combined = combined[(combined.domain != "__none__") & combined.template.ne("")].reset_index(drop=True)
    if combined.label.nunique() != 2:
        raise RuntimeError("Showcase development corpus lost a class during screening")

    groups = component_groups(combined)
    # First reserve 15% grouped test. Then reserve 15/85 of the remainder as
    # validation, yielding approximate 70/15/15 splits with a fixed seed.
    split_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_index, test_index = next(split_test.split(combined, combined.label, groups=groups))
    train_val = combined.iloc[train_val_index].copy().reset_index(drop=True)
    test = combined.iloc[test_index].copy().reset_index(drop=True)
    tv_groups = groups[train_val_index]
    split_val = GroupShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=SEED + 1)
    train_index, val_index = next(split_val.split(train_val, train_val.label, groups=tv_groups))
    train = train_val.iloc[train_index].copy().reset_index(drop=True)
    validation = train_val.iloc[val_index].copy().reset_index(drop=True)

    overlaps = {
        "train_validation": pair_overlap(train, validation),
        "train_test": pair_overlap(train, test),
        "validation_test": pair_overlap(validation, test),
    }
    if any(value for comparison in overlaps.values() for value in comparison.values()):
        raise RuntimeError(f"Showcase split violates an isolation rule: {overlaps}")

    selected_columns = ["url", "label", "source"]
    atomic_csv(train[selected_columns], DATA / "showcase_url_train.csv")
    atomic_csv(validation[selected_columns], DATA / "showcase_url_validation.csv")
    atomic_csv(test[selected_columns], DATA / "showcase_url_test.csv")
    component_sizes = pd.Series(groups).value_counts()
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Final showcase URL development corpus; CompPhish is development data and P9 is preserved historical evidence.",
        "inputs": {
            "development_sources": ["PHIUSIL", "MURL", "CompPhish"],
            "compphish_frozen_path": str(FROZEN.relative_to(ROOT)),
            "compphish_frozen_sha256": sha256(FROZEN),
            "compphish_development_copy": str(COMP_DEV_COPY.relative_to(ROOT)),
            "compphish_development_copy_sha256": sha256(COMP_DEV_COPY),
            "p9_historical_outputs_preserved": True,
            "murl_rows_after_conflict_screening_before_cap": int(murl_before_cap),
            **conflict_info,
            "murl_cap_per_class": MURL_CAP_PER_CLASS,
        },
        "deduplication_and_exclusions": {
            "raw_rows_before_normalized_conflict_exclusion": int(before_norm_conflicts),
            "normalized_label_conflict_keys_excluded": int(len(conflicting_normalized)),
            "rows_excluded_for_normalized_label_conflict": int(before_norm_conflicts - len(combined) - (before_dedup - len(combined))),
            "normalized_duplicates_removed_after_conflict_screening": int(before_dedup - len(combined)),
            "malformed_rows_excluded": int(len(malformed)),
            "deduplication_rule": "keep first by normalized URL after source priority CompPhish, PHIUSIL, MURL; only after excluding conflicting labels",
        },
        "grouping": {
            "registered_domain": "offline public suffix extraction",
            "template": "project url_template: lowercase, remove scheme/query/fragment/trailing slash, mask digit runs",
            "components": int(len(component_sizes)),
            "largest_component_rows": int(component_sizes.max()),
            "p95_component_rows": float(component_sizes.quantile(0.95)),
        },
        "splits": {"seed": SEED, "train": summary(train), "validation": summary(validation), "test": summary(test)},
        "overlap": overlaps,
        "outputs": {
            name: {"path": str((DATA / filename).relative_to(ROOT)), "sha256": sha256(DATA / filename)}
            for name, filename in (("train", "showcase_url_train.csv"), ("validation", "showcase_url_validation.csv"), ("test", "showcase_url_test.csv"))
        },
    }
    (REPORTS / "showcase_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
