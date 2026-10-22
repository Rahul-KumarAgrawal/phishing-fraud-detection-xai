"""
feature_subsets.py
==================
Named URL feature subsets, and the subset chosen for the revised model.

The split here is *derived from measurement*, not intuition. Comparing the mean
feature values of PHIUSIL training data against those of an independent
benchmark (``reports/url_feature_distributions.csv``, produced by
``scripts/url_feature_diagnosis.py``) shows two groups:

* ``FORMAT_SHORTCUT_FEATURES`` -- features whose direction *reverses* between
  datasets: legitimate URLs score higher in one corpus and lower in the other.
  In PHIUSIL every legitimate URL is a bare ``https://domain``, so these mostly
  encode URL *formatting* rather than phishing intent.
* ``CONSISTENT_FEATURES`` -- features whose direction holds in both corpora, plus
  the brand-impersonation features.

Which subset is actually used is decided by running
``scripts/ablate_url_features.py`` and reading the results; see
``reports/url_feature_ablation.md``.

The measured reason these subsets exist: in PHIUSIL every legitimate URL is a
bare ``https://domain`` with no path, while its phishing URLs carry paths, and
the only path-bearing legitimate source used for augmentation (MURL) stores no
scheme at all. The conjunction *"https scheme AND a non-empty path"* therefore
occurred **only in the phishing class** in training data - a rule a tree model
learns exactly, and the direct cause of the 100% false-positive rate on
independent legitimate URLs that do have paths.
"""

from fraud_phishing.features.url_features import FEATURE_NAMES

# Direction is inconsistent between PHIUSIL and the independent benchmark.
FORMAT_SHORTCUT_FEATURES: list[str] = [
    "path_length", "slash_count", "path_depth",
    "query_length", "question_count", "equals_count", "ampersand_count",
    "query_param_count", "has_query",
    "url_length", "hostname_length", "subdomain_length", "subdomain_count",
    "hyphen_count", "underscore_count", "percent_count",
    "suspicious_token_count",
]

# Direction is consistent between both corpora (plus the brand features).
CONSISTENT_FEATURES: list[str] = [
    "registered_domain_length", "dot_count", "digit_count", "digit_ratio",
    "tld_length", "at_count", "tilde_count", "special_char_ratio",
    "is_https", "is_ip_host", "has_at_symbol", "has_percent_encoding",
    "is_url_shortener",
    "brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain",
]

BRAND_FEATURES: list[str] = [
    "brand_token_count", "brand_domain_mismatch", "brand_lookalike_domain",
]

# The two groups must partition the schema exactly; if a new feature is added to
# url_features.py and not classified here, this fires instead of silently
# leaving it out of every subset.
_unclassified = (set(FEATURE_NAMES)
                 - set(FORMAT_SHORTCUT_FEATURES)
                 - set(CONSISTENT_FEATURES))
if _unclassified:
    raise RuntimeError(
        "New URL features are not classified in feature_subsets.py: "
        f"{sorted(_unclassified)}. Add them to FORMAT_SHORTCUT_FEATURES or "
        "CONSISTENT_FEATURES (with evidence from reports/url_feature_distributions.csv)."
    )
_overlap = set(FORMAT_SHORTCUT_FEATURES) & set(CONSISTENT_FEATURES)
if _overlap:
    raise RuntimeError(f"Features classified in both groups: {sorted(_overlap)}")


def _without(*names: str) -> list[str]:
    drop = set(names)
    return [f for f in FEATURE_NAMES if f not in drop]


# Groups used to build the targeted subsets below.
_PATH_FEATURES = ("path_length", "slash_count", "path_depth")
_QUERY_FEATURES = (
    "query_length", "question_count", "equals_count", "ampersand_count",
    "query_param_count", "has_query",
)
_LENGTH_FEATURES = ("url_length", "hostname_length", "subdomain_length",
                    "subdomain_count")

SUBSETS: dict[str, list[str]] = {
    "all": list(FEATURE_NAMES),
    "no_path_features": _without(*_PATH_FEATURES),
    "no_path_no_query": _without(*_PATH_FEATURES, *_QUERY_FEATURES),
    "no_path_no_length": _without(*_PATH_FEATURES, *_LENGTH_FEATURES),
    "no_path_query_length": _without(
        *_PATH_FEATURES, *_QUERY_FEATURES, *_LENGTH_FEATURES
    ),
    "no_shape_features": _without(
        "url_length", "hostname_length", "path_length", "query_length",
        "subdomain_length", "subdomain_count", "path_depth", "slash_count",
        "hyphen_count", "underscore_count",
    ),
    "no_https": _without("is_https"),
    "no_brand": _without(*BRAND_FEATURES),
    "consistent_only": list(CONSISTENT_FEATURES),
    "consistent_no_https": [f for f in CONSISTENT_FEATURES if f != "is_https"],
}

# ── The subset the revised model is trained with ─────────────────────────────
# Set from the ablation results in reports/url_feature_ablation.md.
# Chosen model: LogisticRegression on this subset.
#
# Rationale, from the measured table (augmented pool):
#   * the root cause is that "https AND a non-empty path" occurred only in the
#     phishing class, so path-encoding features must go;
#   * `no_path_no_query` keeps the best PHIUSIL-template F1 (0.9429) of any
#     candidate that fixes the false-positive rate, with 96.3% recall on
#     byte-distinct external phishing URLs;
#   * dropping the features that let a tree express the conjunction matters more
#     than dropping `is_https` - every variant that removed `is_https` made
#     things worse.
REVISED_SUBSET_NAME = "no_path_no_query"
REVISED_MODEL_KIND = "LogisticRegression"
REVISED_FEATURES: list[str] = list(SUBSETS[REVISED_SUBSET_NAME])
