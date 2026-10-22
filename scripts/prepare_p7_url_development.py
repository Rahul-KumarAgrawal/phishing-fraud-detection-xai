"""Prepare deterministic P7 development splits from PHIUSIL plus MURL.

CompPhish, LegitPhish and URL-Phish are never read here. No network or model I/O.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_compphish_overlap import normalized_url  # noqa: E402
from audit_url_datasets import SOURCES, _registered_domain, load_source  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from fraud_phishing.data.prepare_urls import url_template  # noqa: E402

OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "p7"
SEED = 42
MURL_CAP_PER_CLASS = 60_000


class UnionFind:
    def __init__(self, n: int): self.parent = list(range(n)); self.size = [1] * n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]: a, b = b, a
            self.parent[b] = a; self.size[a] += self.size[b]


def _connect_groups(frame: pd.DataFrame) -> np.ndarray:
    uf = UnionFind(len(frame)); seen: dict[tuple[str, str], int] = {}
    for idx, (domain, template) in enumerate(zip(frame["domain"], frame["template"])):
        for kind, value in (("domain", domain), ("template", template)):
            key = (kind, value)
            if key in seen: uf.union(idx, seen[key])
            else: seen[key] = idx
    return np.array([uf.find(i) for i in range(len(frame))])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    phiusil = load_source("PHIUSIL", SOURCES["PHIUSIL"], [])
    phiusil["source"] = "PHIUSIL"
    murl = load_source("MURL", SOURCES["MURL"], [])
    murl["source"] = "MURL"

    # Quarantine MURL strings with a known contradictory exact label in the two
    # development diagnostics. Their rows are not included in fitting.
    conflict_urls: set[str] = set()
    for name in ("LegitPhish", "URL-Phish"):
        other = load_source(name, SOURCES[name], [])
        labels = dict(zip(other["url"], other["label"]))
        conflict_urls |= {u for u, y in zip(murl["url"], murl["label"]) if u in labels and labels[u] != y}
    murl_before_conflicts = len(murl)
    murl = murl[~murl["url"].isin(conflict_urls)].copy()

    rng = np.random.default_rng(SEED)
    capped = []
    for label in (0, 1):
        part = murl[murl["label"] == label]
        if len(part) > MURL_CAP_PER_CLASS:
            part = part.iloc[np.sort(rng.choice(len(part), MURL_CAP_PER_CLASS, replace=False))]
        capped.append(part)
    murl = pd.concat(capped, ignore_index=True)
    combined = pd.concat([phiusil, murl], ignore_index=True)
    combined["normalized"] = combined["url"].map(normalized_url)
    # Conflicting labels after conservative normalisation are excluded rather
    # than assigning a preferred source label.
    norm_labels = combined.groupby("normalized")["label"].nunique()
    bad_norm = set(norm_labels[norm_labels > 1].index)
    combined = combined[~combined["normalized"].isin(bad_norm)].copy()
    combined = combined.sort_values(["source", "url"], kind="stable").drop_duplicates("normalized", keep="first")
    combined["domain"] = combined["url"].map(_registered_domain)
    combined["template"] = combined["url"].map(url_template)
    combined = combined[(combined["domain"] != "__none__") & combined["template"].ne("")].reset_index(drop=True)
    groups = _connect_groups(combined)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_i, val_i = next(gss.split(combined, combined["label"], groups=groups))
    train, val = combined.iloc[train_i].copy(), combined.iloc[val_i].copy()
    for field in ("url", "normalized", "domain", "template"):
        overlap = set(train[field]) & set(val[field])
        if overlap:
            raise RuntimeError(f"P7 split leaks {len(overlap)} {field} key(s)")
    train[["url", "label", "source"]].to_csv(OUT / "p7_url_train.csv", index=False)
    val[["url", "label", "source"]].to_csv(OUT / "p7_url_val.csv", index=False)
    group_sizes = pd.Series(groups).value_counts()
    stats = {
        "seed": SEED,
        "development_boundary": ["PHIUSIL", "MURL"],
        "not_read": ["compphish_external_frozen.csv", "LegitPhish", "URL-Phish"],
        "murl_exact_label_conflicts_quarantined": murl_before_conflicts - len(murl) if False else len(conflict_urls),
        "murl_rows_after_conflict_quarantine_before_cap": murl_before_conflicts - len(conflict_urls),
        "murl_cap_per_class": MURL_CAP_PER_CLASS,
        "normalized_label_conflict_keys_excluded": len(bad_norm),
        "grouping": "connected components where any shared registered domain or project-normalized template joins rows",
        "components": {"count": int(len(group_sizes)), "largest_rows": int(group_sizes.max()), "p95_rows": float(group_sizes.quantile(.95))},
        "train": _summary(train), "validation": _summary(val),
        "isolation": {"exact_url": 0, "normalized_url": 0, "registered_domain": 0, "template": 0},
    }
    (REPORTS / "p7_split_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def _summary(frame: pd.DataFrame) -> dict:
    return {"rows": int(len(frame)), "legitimate": int((frame.label == 0).sum()), "phishing": int((frame.label == 1).sum()), "phiusil": int((frame.source == "PHIUSIL").sum()), "murl": int((frame.source == "MURL").sum())}


if __name__ == "__main__": main()
