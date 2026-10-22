"""
test_url_split_leakage.py
=========================
Guards the URL pipeline against the two silent failure modes that previously
made this project's headline numbers meaningless:

1. **Inverted labels.** PHIUSIL stores ``1 = legitimate`` and ``0 = phishing``
   (the opposite of the intuitive convention). Reading it backwards inverts
   every metric and every label the API returns, while still "looking great".

2. **Evaluation leakage.** The domain-aware test set was scored with a model
   trained on a random split that shared ~85% of its URLs, so the reported
   unseen-domain performance measured memorisation.

Both bugs produced high scores, so neither could be caught by looking at the
numbers. These tests assert the invariants directly.
"""

import json
import pathlib
from urllib.parse import urlsplit

import pandas as pd
import pytest

from fraud_phishing.data.prepare_urls import (
    PHIUSIL_EXPECTED_RAW_COUNTS,
    PHIUSIL_LABEL_MAP_STR,
    normalize_label,
    url_template,
    verify_raw_label_convention,
)

PROCESSED = pathlib.Path("data/processed")
LEAKAGE_REPORT = pathlib.Path("reports/url_leakage_report.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label convention
# ─────────────────────────────────────────────────────────────────────────────

def test_phiusil_raw_labels_map_to_project_convention():
    """Raw 1 is legitimate and raw 0 is phishing -- not the other way round."""
    assert PHIUSIL_LABEL_MAP_STR["1"] == 0, "raw 1 (legitimate) must map to 0"
    assert PHIUSIL_LABEL_MAP_STR["0"] == 1, "raw 0 (phishing) must map to 1"


def test_normalize_label_inverts_phiusil_raw_values():
    assert normalize_label(pd.Series([1, 1, 0, 0])).tolist() == [0, 0, 1, 1]


def test_verify_raw_label_convention_accepts_documented_counts():
    verify_raw_label_convention(pd.Series(PHIUSIL_EXPECTED_RAW_COUNTS))


def test_verify_raw_label_convention_rejects_inverted_counts():
    inverted = pd.Series(
        {
            1: PHIUSIL_EXPECTED_RAW_COUNTS[0],
            0: PHIUSIL_EXPECTED_RAW_COUNTS[1],
        }
    )
    with pytest.raises(ValueError):
        verify_raw_label_convention(inverted)


def test_verify_raw_label_convention_rejects_unknown_composition():
    with pytest.raises(ValueError):
        verify_raw_label_convention(pd.Series({1: 10, 0: 10}))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Template helper
# ─────────────────────────────────────────────────────────────────────────────

def test_url_template_collapses_volatile_parts():
    a = url_template("https://Shop.Example.com/Product/12345?ref=abc#frag")
    b = url_template("http://shop.example.com/Product/67890")
    assert a == b


def test_url_template_keeps_distinct_paths_apart():
    assert url_template("https://a.example.com/login") != url_template(
        "https://a.example.com/verify"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split integrity (runs only when the pipeline output is present)
# ─────────────────────────────────────────────────────────────────────────────

def _load(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path} not found; run prepare_urls.py first")
    return pd.read_csv(path)


def _registered_domains(urls) -> set[str]:
    """Registered domains, computed only for unique hosts to stay fast."""
    hosts = set()
    for u in urls:
        try:
            host = urlsplit(u if "//" in u else f"//{u}").hostname
        except ValueError:
            continue
        if host:
            hosts.add(host.lower())

    import tldextract

    return {
        d
        for d in (
            tldextract.extract(h).top_domain_under_public_suffix for h in hosts
        )
        if d
    }


def test_random_splits_are_disjoint_on_exact_url():
    train = set(_load("urls_train.csv")["url"])
    val = set(_load("urls_val.csv")["url"])
    test = set(_load("urls_test.csv")["url"])

    assert train & val == set(), "exact URL shared between train and val"
    assert train & test == set(), "exact URL shared between train and test"
    assert val & test == set(), "exact URL shared between val and test"


def test_domain_split_shares_no_registered_domain():
    train_urls = _load("urls_domain_train.csv")["url"].tolist()
    test_urls = _load("urls_domain_test.csv")["url"].tolist()

    assert set(train_urls) & set(test_urls) == set(), "exact URL overlap"
    overlap = _registered_domains(train_urls) & _registered_domains(test_urls)
    assert overlap == set(), f"shared registered domains: {sorted(overlap)[:10]}"


def test_template_clean_split_shares_no_url_template():
    train_templates = {url_template(u) for u in _load("urls_template_train.csv")["url"]}
    test_urls = _load("urls_template_test.csv")["url"].tolist()

    shared = [u for u in test_urls if url_template(u) in train_templates]
    assert shared == [], f"{len(shared)} test rows share a training template: {shared[:3]}"


def test_leakage_report_records_the_legacy_contamination():
    """
    The report must not claim the old random-split evaluation was clean: it had
    tens of thousands of shared URLs. This pins that fact so the corrected
    numbers can always be traced back to why they differ.
    """
    if not LEAKAGE_REPORT.exists():
        pytest.skip(f"{LEAKAGE_REPORT} not found; run prepare_urls.py first")

    reports = json.loads(LEAKAGE_REPORT.read_text(encoding="utf-8"))["reports"]
    by_name = {r["split"]: r for r in reports}

    legacy = next(
        (r for r in reports if r["split"].startswith("LEGACY")),
        None,
    )
    assert legacy is not None, "legacy contaminated split must be reported"
    assert legacy["exact_url_overlap"] > 0

    domain = by_name.get("domain_train vs domain_test")
    assert domain is not None
    assert domain["exact_url_overlap"] == 0
    assert domain["domain_overlap"] == 0

    clean = by_name.get("domain_train vs template_clean_test")
    assert clean is not None
    assert clean["test_template_overlap_count"] == 0
