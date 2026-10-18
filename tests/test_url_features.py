"""
test_url_features.py
====================
Unit tests for the canonical URL feature extractor.

Harmless synthetic strings and reserved example domains only.
No actual phishing URLs are navigated.
"""

import math
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fraud_phishing.features.url_features import (
    FEATURE_NAMES,
    SUSPICIOUS_TOKENS,
    URL_SHORTENERS,
    extract_features,
    extract_feature_vector,
    extract_urls_from_text,
    _is_ip_address,
)


# ── Schema ───────────────────────────────────────────────────────────────────

def test_feature_names_are_tuple():
    assert isinstance(FEATURE_NAMES, tuple)


def test_feature_count():
    """Feature schema must have exactly 30 features."""
    assert len(FEATURE_NAMES) == 30


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_extract_features_returns_all_keys():
    result = extract_features("https://example.com/path?q=1")
    assert set(result.keys()) == set(FEATURE_NAMES)


def test_feature_vector_length():
    vec = extract_feature_vector("http://example.com")
    assert len(vec) == len(FEATURE_NAMES)


def test_feature_vector_all_finite():
    vec = extract_feature_vector("http://example.com/login?user=test&pass=abc")
    assert all(math.isfinite(v) for v in vec)


# ── HTTPS indicator ──────────────────────────────────────────────────────────

def test_is_https_true():
    f = extract_features("https://secure.example.com/")
    assert f["is_https"] == 1


def test_is_https_false():
    f = extract_features("http://example.com/")
    assert f["is_https"] == 0


# ── IP address detection ──────────────────────────────────────────────────────

def test_ip_host_ipv4():
    f = extract_features("http://192.168.1.1/login")
    assert f["is_ip_host"] == 1


def test_ip_host_ipv6():
    assert _is_ip_address("::1") == 1
    assert _is_ip_address("[::1]") == 1


def test_ip_host_not_for_domain():
    f = extract_features("http://example.com/")
    assert f["is_ip_host"] == 0


# ── Length features ───────────────────────────────────────────────────────────

def test_url_length():
    url = "http://example.com/path"
    f = extract_features(url)
    assert f["url_length"] == len(url)


def test_path_length():
    f = extract_features("http://example.com/this/is/a/path")
    assert f["path_length"] == len("/this/is/a/path")


def test_query_length():
    f = extract_features("http://example.com/?foo=bar")
    assert f["query_length"] == len("foo=bar")


# ── Character counts ──────────────────────────────────────────────────────────

def test_dot_count():
    f = extract_features("http://sub.sub2.example.com/")
    assert f["dot_count"] == "http://sub.sub2.example.com/".count(".")


def test_hyphen_count():
    f = extract_features("http://my-site.example-test.com/")
    assert f["hyphen_count"] == 2


def test_at_count_and_has_at():
    f = extract_features("http://user@example.com/")
    assert f["at_count"] >= 1
    assert f["has_at_symbol"] == 1


def test_percent_encoding():
    f = extract_features("http://example.com/%20path%21")
    assert f["percent_count"] >= 2
    assert f["has_percent_encoding"] == 1


def test_digit_ratio_range():
    f = extract_features("http://example123.com/page456")
    assert 0.0 <= f["digit_ratio"] <= 1.0


# ── Domain structure ──────────────────────────────────────────────────────────

def test_subdomain_count():
    f = extract_features("http://sub.sub2.example.com/")
    assert f["subdomain_count"] == 2


def test_no_subdomain():
    f = extract_features("http://example.com/")
    assert f["subdomain_count"] == 0


def test_path_depth():
    f = extract_features("http://example.com/a/b/c/")
    assert f["path_depth"] == 3


# ── Query parameters ──────────────────────────────────────────────────────────

def test_has_query_true():
    f = extract_features("http://example.com/?key=val")
    assert f["has_query"] == 1
    assert f["query_param_count"] >= 1


def test_has_query_false():
    f = extract_features("http://example.com/page")
    assert f["has_query"] == 0
    assert f["query_param_count"] == 0


# ── URL shortener ─────────────────────────────────────────────────────────────

def test_url_shortener_detected():
    f = extract_features("http://bit.ly/abc123")
    assert f["is_url_shortener"] == 1


def test_url_shortener_not_detected():
    f = extract_features("http://example.com/abc123")
    assert f["is_url_shortener"] == 0


# ── Suspicious tokens ─────────────────────────────────────────────────────────

def test_suspicious_token_count_zero():
    f = extract_features("http://example.com/about-us")
    assert f["suspicious_token_count"] == 0


def test_suspicious_token_count_positive():
    f = extract_features("http://example.com/login?verify=true")
    assert f["suspicious_token_count"] >= 2  # "login" and "verify"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_missing_scheme():
    """URLs without a scheme should be handled gracefully."""
    f = extract_features("example.com/path")
    assert isinstance(f["url_length"], int)


def test_empty_url_raises():
    with pytest.raises(ValueError):
        extract_features("")


def test_non_string_raises():
    with pytest.raises(ValueError):
        extract_features(None)


def test_excessively_long_url_truncated():
    long_url = "http://example.com/" + "a" * 5000
    f = extract_features(long_url)
    assert f["url_length"] <= 4096


def test_unicode_domain():
    """Punycode / IDN domains should not raise."""
    f = extract_features("http://例え.jp/path")
    assert isinstance(f["url_length"], int)


def test_port_in_url():
    f = extract_features("http://example.com:8080/path")
    assert f["hostname_length"] > 0


def test_ftp_scheme():
    f = extract_features("ftp://example.com/file.txt")
    assert f["is_https"] == 0


def test_determinism():
    """Same URL must always produce the same features."""
    url = "https://sub.example.com/path?a=1&b=2#frag"
    assert extract_features(url) == extract_features(url)


# ── URL extraction from text ──────────────────────────────────────────────────

def test_extract_urls_from_text_empty():
    assert extract_urls_from_text("No URLs here.") == []


def test_extract_urls_from_text_finds_http():
    urls = extract_urls_from_text("Click here: http://example.com/promo")
    assert any("example.com" in u for u in urls)


def test_extract_urls_from_text_finds_www():
    urls = extract_urls_from_text("Visit www.example.com for details!")
    assert any("example.com" in u for u in urls)


def test_extract_urls_no_duplicates():
    text = "http://example.com http://example.com"
    urls = extract_urls_from_text(text)
    assert len(urls) == len(set(urls))
