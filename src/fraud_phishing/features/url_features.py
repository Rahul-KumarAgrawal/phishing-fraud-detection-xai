"""
url_features.py
===============
Canonical static URL feature extractor.

This is the single source of truth for URL features used in both
training and API inference. The feature schema (name and order) is
immutable – any change requires retraining the URL model.

Features (28 total):
  Group 1 – Length & shape (5)
  Group 2 – Character counts (14)
  Group 3 – Domain structure (3)
  Group 4 – URL components (3)
  Group 5 – Boolean / lexical indicators (6)
  → Suspicious-token count uses an explicit, documented token list.

Usage:
    from fraud_phishing.features.url_features import extract_features, FEATURE_NAMES
    features = extract_features("https://example.com/login")
    # Returns: dict[str, float | int]

No network I/O is performed. The URL destination is never visited.
"""

import ipaddress
import re
import urllib.parse
from typing import Any

import tldextract

# ─────────────────────────────────────────────────────────────────────────────
# Immutable feature schema (name list defines order used by the model)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: tuple[str, ...] = (
    # Group 1 – Length & shape
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "registered_domain_length",
    # Group 2 – Character counts
    "dot_count",
    "hyphen_count",
    "slash_count",
    "underscore_count",
    "at_count",
    "question_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "digit_count",
    "digit_ratio",
    "special_char_ratio",
    "tilde_count",
    # Group 3 – Domain structure
    "subdomain_count",
    "subdomain_length",
    "tld_length",
    # Group 4 – URL components
    "path_depth",
    "query_param_count",
    "has_query",
    # Group 5 – Boolean / lexical indicators
    "is_https",
    "is_ip_host",
    "has_at_symbol",
    "has_percent_encoding",
    "is_url_shortener",
    "suspicious_token_count",
)

# Documented list of suspicious tokens used for the suspicious_token_count feature.
# Note: presence of these tokens is a risk *indicator*, not proof of phishing.
# Legitimate URLs can contain any of these tokens.
SUSPICIOUS_TOKENS: frozenset[str] = frozenset(
    {
        "login",
        "signin",
        "sign-in",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "banking",
        "confirm",
        "password",
        "credential",
        "wallet",
        "paypal",
        "apple",
        "google",
        "microsoft",
        "amazon",
        "ebay",
        "support",
        "helpdesk",
        "click",
        "free",
        "prize",
        "winner",
        "lucky",
        "claim",
        "limited",
        "offer",
        "gift",
        "redeem",
        "urgent",
        "alert",
        "warning",
    }
)

# Known URL shortener domains
URL_SHORTENERS: frozenset[str] = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bitly.com",
        "rebrandly.com",
        "cutt.ly",
        "short.io",
        "rb.gy",
        "shorturl.at",
        "url.ie",
        "lnkd.in",
    }
)

# Regex for special characters (non-alphanumeric, excluding common separators)
_SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9./\-_~:?#\[\]@!$&'()*+,;=%]")
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+\-.]*://", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Core extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(url: str) -> dict[str, Any]:
    """
    Extract a fixed set of numeric/boolean features from a URL string.

    Parameters
    ----------
    url : str
        Raw URL string. Must not be empty. No network I/O is performed.

    Returns
    -------
    dict[str, Any]
        Keys match FEATURE_NAMES exactly. Values are int or float.

    Raises
    ------
    ValueError
        If `url` is empty or not a string.
    """
    if not isinstance(url, str):
        raise ValueError(f"url must be a str, got {type(url).__name__}")
    url = url.strip()
    if not url:
        raise ValueError("url must not be empty")

    # Truncate excessively long URLs (> 4096 chars) to avoid regex DoS
    MAX_URL_LEN = 4096
    if len(url) > MAX_URL_LEN:
        url = url[:MAX_URL_LEN]

    # Add scheme if missing so urllib.parse can split components
    working_url = url
    if not _URL_SCHEME_RE.match(url):
        working_url = "http://" + url

    try:
        parsed = urllib.parse.urlparse(working_url)
    except Exception:
        parsed = urllib.parse.ParseResult("", "", "", "", "", "")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # tldextract (uses offline public suffix list; no network I/O)
    try:
        ext = tldextract.extract(working_url)
        # top_domain_under_public_suffix is the current API (tldextract >= 5.x)
        registered_domain = ext.top_domain_under_public_suffix or ""
        subdomain = ext.subdomain or ""
        suffix = ext.suffix or ""
    except Exception:
        registered_domain = ""
        subdomain = ""
        suffix = ""

    # ── Group 1 – Length & shape ─────────────────────────────────────────
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)
    registered_domain_length = len(registered_domain)

    # ── Group 2 – Character counts ───────────────────────────────────────
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    slash_count = url.count("/")
    underscore_count = url.count("_")
    at_count = url.count("@")
    question_count = url.count("?")
    equals_count = url.count("=")
    ampersand_count = url.count("&")
    percent_count = url.count("%")
    tilde_count = url.count("~")

    digits = sum(c.isdigit() for c in url)
    digit_count = digits
    digit_ratio = digits / url_length if url_length > 0 else 0.0
    special_chars = len(_SPECIAL_CHAR_RE.findall(url))
    special_char_ratio = special_chars / url_length if url_length > 0 else 0.0

    # ── Group 3 – Domain structure ───────────────────────────────────────
    subdomain_parts = [s for s in subdomain.split(".") if s] if subdomain else []
    subdomain_count = len(subdomain_parts)
    subdomain_length = len(subdomain)
    tld_length = len(suffix)

    # ── Group 4 – URL components ─────────────────────────────────────────
    path_segments = [p for p in path.split("/") if p]
    path_depth = len(path_segments)
    query_params = urllib.parse.parse_qs(query, keep_blank_values=True)
    query_param_count = len(query_params)
    has_query = int(bool(query))

    # ── Group 5 – Boolean / lexical indicators ───────────────────────────
    is_https = int(parsed.scheme.lower() == "https")
    is_ip_host = _is_ip_address(hostname)
    has_at_symbol = int(at_count > 0)
    has_percent_encoding = int(percent_count > 0)

    # URL shortener detection (registered domain match)
    regd_lower = registered_domain.lower()
    is_url_shortener = int(
        regd_lower in URL_SHORTENERS
        or any(regd_lower == s for s in URL_SHORTENERS)
    )

    # Suspicious token count (from documented SUSPICIOUS_TOKENS list)
    url_lower = url.lower()
    suspicious_token_count = sum(
        1 for token in SUSPICIOUS_TOKENS if token in url_lower
    )

    features: dict[str, Any] = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "registered_domain_length": registered_domain_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "slash_count": slash_count,
        "underscore_count": underscore_count,
        "at_count": at_count,
        "question_count": question_count,
        "equals_count": equals_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "special_char_ratio": special_char_ratio,
        "tilde_count": tilde_count,
        "subdomain_count": subdomain_count,
        "subdomain_length": subdomain_length,
        "tld_length": tld_length,
        "path_depth": path_depth,
        "query_param_count": query_param_count,
        "has_query": has_query,
        "is_https": is_https,
        "is_ip_host": is_ip_host,
        "has_at_symbol": has_at_symbol,
        "has_percent_encoding": has_percent_encoding,
        "is_url_shortener": is_url_shortener,
        "suspicious_token_count": suspicious_token_count,
    }

    # ── Schema validation ────────────────────────────────────────────────
    missing = set(FEATURE_NAMES) - set(features)
    extra = set(features) - set(FEATURE_NAMES)
    if missing:
        raise RuntimeError(f"Feature extraction produced missing features: {missing}")
    if extra:
        raise RuntimeError(f"Feature extraction produced unexpected features: {extra}")

    return features


def extract_feature_vector(url: str) -> list[float]:
    """
    Return features as an ordered list matching FEATURE_NAMES.
    Suitable for passing directly to a trained scikit-learn model.
    """
    d = extract_features(url)
    return [float(d[name]) for name in FEATURE_NAMES]


def _is_ip_address(hostname: str) -> int:
    """Return 1 if hostname is an IPv4 or IPv6 literal address, else 0."""
    if not hostname:
        return 0
    # Strip IPv6 brackets
    h = hostname.strip("[]")
    try:
        ipaddress.ip_address(h)
        return 1
    except ValueError:
        return 0


def extract_urls_from_text(text: str) -> list[str]:
    """
    Extract URL candidates from a free-text message string.
    Returns a deduplicated list of candidate URLs found in the text.
    No network I/O is performed.
    """
    pattern = re.compile(
        r"(?:https?://|www\.)"          # scheme or www
        r"[^\s\"'<>(){}\[\]\\,;]+"      # non-whitespace URL body
        r"(?<![.,!?;:'\")])",            # strip trailing punctuation
        re.IGNORECASE,
    )
    found = pattern.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for u in found:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result
