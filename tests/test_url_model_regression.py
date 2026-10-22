"""
test_url_model_regression.py
============================
End-to-end regression tests for the URL classifier.

These lock in the behaviour that the P3 work fixed. Before it, every legitimate
URL that carried a path was classified as phishing (100% false-positive rate on
the 445 byte-distinct legitimate URLs in the LegitPhish benchmark) because every
legitimate URL in PHIUSIL is a bare ``https://domain``. See
``reports/url_diagnosis.md``.

The tests run against whichever model artifacts exist and are skipped otherwise,
so the feature-level suite still passes on a clean checkout.

Only synthetic / reserved-example strings are used; no real phishing URL is
visited (and the extractor never performs network I/O anyway).
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.models.url_model import (  # noqa: E402
    URL_MODEL_PATH,
    URL_REVISED_MODEL_PATH,
    load_model,
    load_revised_model,
    predict_url,
)

# ── Curated cases ────────────────────────────────────────────────────────────

LEGITIMATE_WITH_PATH_OR_QUERY = [
    "https://www.youtube.com/watch?v=eP_Ylb5VE9M",
    "https://stackoverflow.com/questions/tagged/php",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://github.com/psf/requests/issues/1234",
    "https://www.google.com/search?q=fraud+detection&hl=en",
]

# A known, measured limitation of the revised model: a long legitimate URL whose
# path is full of digits (here a publication date) still looks phishing-like.
# It is recorded as a documented xfail rather than dropped from the suite.
LEGITIMATE_DIGIT_HEAVY_PATH = [
    "https://www.forbes.com/sites/forbestechcouncil/2024/01/01/example/",
]

LEGITIMATE_BARE_HTTPS = [
    "https://www.example.com",
    "https://www.google.com",
]

# Unambiguous phishing shapes rather than borderline ones.
PHISHING_IP_HOST = [
    "http://103.15.254.56:33/login",
    "http://192.168.14.9/secure/verify.php",
]

PHISHING_BRAND_IMPERSONATION = [
    "https://paypal-secure-login.verify-account.com/login",
    "https://secure-apple-id-locked.support-apple.com/signin",
]


def _revised_variant():
    """
    The revised model, from whichever artifact holds it.

    Before the switch is approved the revised model lives in
    ``url_model_revised.joblib``; afterwards it becomes the served
    ``url_model.joblib``. Whichever carries ``model_variant == "revised"`` is used,
    so these tests follow the model automatically once it is promoted.
    """
    for loader, path in ((load_model, URL_MODEL_PATH),
                         (load_revised_model, URL_REVISED_MODEL_PATH)):
        if not path.exists():
            continue
        try:
            pipe, meta = loader()
        except Exception:  # noqa: BLE001 - a stale artifact should skip, not error
            continue
        if meta.get("model_variant") == "revised":
            return pipe, meta
    return None


def _predict_all(urls):
    """Yield (url, result) for the selected revised model."""
    variant = _revised_variant()
    if variant is None:
        pytest.skip("No revised URL model artifact present.")
    pipe, meta = variant
    feats = meta.get("feature_names")
    thr = float(meta.get("threshold", 0.5))
    for url in urls:
        yield url, predict_url(url, pipe, thr, feature_names=feats)


# ── The bug that was fixed ───────────────────────────────────────────────────

def test_metadata_declares_matching_feature_names():
    """The saved feature list must line up with the live extractor."""
    from fraud_phishing.features.url_features import FEATURE_NAMES  # noqa: PLC0415

    variant = _revised_variant()
    if variant is None:
        pytest.skip("No revised URL model artifact present.")
    _, meta = variant
    feats = meta.get("feature_names")
    assert feats, "revised metadata has no feature_names"
    assert all(f in FEATURE_NAMES for f in feats)
    positions = [FEATURE_NAMES.index(f) for f in feats]
    assert positions == sorted(positions)


@pytest.mark.parametrize("url", LEGITIMATE_WITH_PATH_OR_QUERY)
def test_legitimate_urls_with_paths_are_not_flagged(url):
    for u, result in _predict_all([url]):
        assert result["label"] == 0, (
            f"revised model flagged a legitimate path-bearing URL as phishing: "
            f"{u} -> {result['score']:.4f}"
        )


@pytest.mark.parametrize("url", LEGITIMATE_BARE_HTTPS)
def test_legitimate_bare_https_urls_are_not_flagged(url):
    for u, result in _predict_all([url]):
        assert result["label"] == 0, f"{u} -> {result['score']:.4f}"


@pytest.mark.parametrize("url", LEGITIMATE_DIGIT_HEAVY_PATH)
@pytest.mark.xfail(
    reason=(
        "Measured limitation of the revised model: a long legitimate URL whose "
        "path is full of digits still scores slightly above the threshold."
    ),
    strict=False,
)
def test_known_limitation_digit_heavy_paths(url):
    """Recorded so the suite states the weakness instead of dropping the case."""
    for u, result in _predict_all([url]):
        assert result["label"] == 0, (
            f"revised model flagged a digit-heavy legitimate path: "
            f"{u} -> {result['score']:.4f}"
        )


@pytest.mark.parametrize("url", PHISHING_IP_HOST)
def test_ip_host_phishing_is_flagged(url):
    for u, result in _predict_all([url]):
        assert result["label"] == 1, (
            f"revised model missed an IP-host phishing URL: {u} -> {result['score']:.4f}"
        )


@pytest.mark.parametrize("url", PHISHING_BRAND_IMPERSONATION)
def test_brand_impersonation_is_flagged(url):
    for u, result in _predict_all([url]):
        assert result["label"] == 1, (
            f"revised model missed brand impersonation: {u} -> {result['score']:.4f}"
        )


def test_probability_is_the_phishing_class():
    """score must be P(phishing), not P(legitimate) and not a raw decision value."""
    variant = _revised_variant()
    if variant is None:
        pytest.skip("No revised URL model artifact present.")
    pipe, meta = variant
    feats = meta.get("feature_names")
    thr = float(meta.get("threshold", 0.5))
    legit = predict_url("https://www.google.com", pipe, thr, feature_names=feats)
    phish = predict_url("http://103.15.254.56:33/login", pipe, thr, feature_names=feats)
    assert 0.0 <= legit["score"] <= 1.0
    assert 0.0 <= phish["score"] <= 1.0
    assert phish["score"] > legit["score"]
    assert legit["label_name"] == "legitimate"
    assert phish["label_name"] == "phishing"
