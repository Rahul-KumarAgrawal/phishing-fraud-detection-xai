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
    BRAND_TOKENS,
    FEATURE_NAMES,
    SUSPICIOUS_TOKENS,
    URL_SHORTENERS,
    extract_features,
    extract_feature_vector,
    extract_urls_from_text,
    _is_ip_address,
    _levenshtein,
    normalize_url_for_parsing,
)


# ── Schema ───────────────────────────────────────────────────────────────────

def test_feature_names_are_tuple():
    assert isinstance(FEATURE_NAMES, tuple)


def test_feature_count():
    """Feature schema must have exactly 33 features."""
    assert len(FEATURE_NAMES) == 33


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_docstring_feature_counts_match_the_schema():
    """
    The module docstring is the first thing a reader sees, and it was wrong once
    (it claimed 28 features, with Group 2 = 14). Pin it to the code.

    P5 verification: the numbers were already correct; this test is the guard
    that keeps them correct, including the per-group breakdown.
    """
    import re

    import fraud_phishing.features.url_features as url_features

    doc = url_features.__doc__ or ""
    total = re.search(r"Features \((\d+) total\)", doc)
    assert total, "docstring no longer states the feature total"
    assert int(total.group(1)) == len(FEATURE_NAMES)

    documented = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(r"Group (\d+) [^\n]*?\((\d+)\)", doc)
    }
    assert documented, "docstring no longer lists per-group counts"
    assert sum(documented.values()) == len(FEATURE_NAMES)

    # Cross-check the docstring against the group markers in the tuple itself.
    source = pathlib.Path(url_features.__file__).read_text(encoding="utf-8")
    body = source.split("FEATURE_NAMES: tuple[str, ...] = (", 1)[1].split("\n)", 1)[0]
    chunks = re.split(r"#\s*Group \d+", body)[1:]
    if chunks and len(chunks) == len(documented):
        counted = [len(re.findall(r'"([a-z0-9_]+)"', chunk)) for chunk in chunks]
        assert counted == [documented[group] for group in sorted(documented)]


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

def test_scheme_less_normalization_is_idempotent_and_does_not_invent_transport():
    normalized = normalize_url_for_parsing("www.example.com/path?q=1")
    assert normalized == "//www.example.com/path?q=1"
    assert normalize_url_for_parsing(normalized) == normalized
    assert normalize_url_for_parsing("https://www.example.com/") == "https://www.example.com/"
    assert normalize_url_for_parsing("http://www.example.com/") == "http://www.example.com/"


def test_scheme_less_url_parses_host_path_and_query_without_claiming_https():
    raw = "www.example.com/path?q=1&lang=en"
    scheme_less = extract_features(raw)
    explicit_http = extract_features(f"http://{raw}")
    explicit_https = extract_features(f"https://{raw}")
    for name in ("hostname_length", "path_length", "query_length", "path_depth", "query_param_count", "has_query", "subdomain_count", "tld_length"):
        assert scheme_less[name] == explicit_http[name] == explicit_https[name]
    assert scheme_less["is_https"] == 0
    assert explicit_http["is_https"] == 0
    assert explicit_https["is_https"] == 1


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


# ── Brand impersonation ───────────────────────────────────────────────────────
#
# These are the cases the original model missed (see reports/url_diagnosis.md):
# it scored "https://www.netflix-vn.com" as legitimate because the URL had no
# path, while flagging every real path-bearing legitimate URL as phishing.

def test_brand_token_count_zero_for_unrelated_domain():
    f = extract_features("https://www.example.com/about")
    assert f["brand_token_count"] == 0
    assert f["brand_domain_mismatch"] == 0
    assert f["brand_lookalike_domain"] == 0


def test_real_brand_domain_is_not_impersonation():
    """The brand's own domain must never look like impersonation."""
    for url in (
        "https://www.paypal.com/signin",
        "https://accounts.google.com/signin",
        "https://www.amazon.com/gp/cart",
    ):
        f = extract_features(url)
        assert f["brand_token_count"] >= 1, url
        assert f["brand_domain_mismatch"] == 0, url
        assert f["brand_lookalike_domain"] == 0, url


def test_brand_in_hostname_of_other_domain_is_mismatch():
    f = extract_features("https://paypal-secure-login.verify-account.com/login")
    assert f["brand_token_count"] == 1
    assert f["brand_domain_mismatch"] == 1


def test_brand_as_subdomain_of_other_domain_is_mismatch():
    f = extract_features("https://paypal.com.evil-site.net/verify")
    assert f["brand_domain_mismatch"] == 1


def test_brand_tokenisation_avoids_false_positive():
    """'applebees' and 'stackoverflow' must not match 'apple' / 'app'."""
    assert extract_features("https://applebees.com/menu")["brand_token_count"] == 0
    assert extract_features("https://stackoverflow.com/questions")["brand_token_count"] == 0


def test_lookalike_domains_detected():
    assert extract_features("https://paypa1.com/login")["brand_lookalike_domain"] == 1
    assert extract_features("https://rnicrosoft.com/account")["brand_lookalike_domain"] == 1


def test_lookalike_not_triggered_for_ordinary_domain():
    assert extract_features("https://example.com/")["brand_lookalike_domain"] == 0
    assert extract_features("https://openai.com/")["brand_lookalike_domain"] == 0


def test_levenshtein_basic():
    assert _levenshtein("paypal", "paypal", 2) == 0
    assert _levenshtein("paypa1", "paypal", 2) == 1
    assert _levenshtein("rnicrosoft", "microsoft", 2) == 2  # r->m and delete n
    assert _levenshtein("abcdef", "uvwxyz", 2) == 3  # bounded: max_distance + 1


def test_brand_tokens_are_lowercase():
    assert all(b == b.lower() for b in BRAND_TOKENS)


# ── Regression: realistic legitimate URL shapes ───────────────────────────────
#
# Every PHIUSIL legitimate URL is a bare "https://domain", so the extractor was
# never exercised on realistic legitimate shapes. These assert that the features
# for such URLs are well-formed and that nothing pathological fires.

_REALISTIC_LEGITIMATE = [
    "https://www.youtube.com/watch?v=eP_Ylb5VE9M",
    "https://stackoverflow.com/questions/tagged/php",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.forbes.com/sites/forbestechcouncil/2024/01/01/example/",
    "https://github.com/psf/requests/issues/1234",
]


@pytest.mark.parametrize("url", _REALISTIC_LEGITIMATE)
def test_realistic_legitimate_urls_extract_cleanly(url):
    f = extract_features(url)
    assert set(f.keys()) == set(FEATURE_NAMES)
    assert all(math.isfinite(float(v)) for v in f.values())
    assert f["is_https"] == 1
    assert f["path_length"] > 0
    assert f["is_ip_host"] == 0
    assert f["has_at_symbol"] == 0


@pytest.mark.parametrize("url", _REALISTIC_LEGITIMATE)
def test_realistic_legitimate_urls_have_no_brand_mismatch(url):
    """None of the brands below appear in these URLs as a third-party token."""
    f = extract_features(url)
    assert f["brand_domain_mismatch"] == 0
    assert f["brand_lookalike_domain"] == 0


def test_query_parameters_do_not_look_phishy_on_their_own():
    """A legitimate search URL must not trip the IP / @ / brand checks."""
    f = extract_features("https://www.google.com/search?q=fraud+detection&hl=en")
    assert f["is_ip_host"] == 0
    assert f["has_at_symbol"] == 0
    assert f["brand_domain_mismatch"] == 0
    assert f["query_param_count"] >= 2
