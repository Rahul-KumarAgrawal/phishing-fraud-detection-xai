"""Focused fixtures for the read-only P6 URL-audit helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SPEC = spec_from_file_location(
    "p6_audit", Path("scripts/audit_url_datasets.py")
)
assert SPEC and SPEC.loader
AUDIT = module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_raw_metrics_preserves_missing_scheme_and_url_components():
    metrics = AUDIT._raw_metrics("shop.example.co.uk/a-7?q=login#part")
    assert metrics["with_path"] == 1
    assert metrics["has_query"] == 1
    assert metrics["digits"] == 1
    assert metrics["hyphens"] == 1


def test_registered_domain_uses_offline_psl_and_normalizes_subdomain():
    assert AUDIT._registered_domain("https://a.b.example.co.uk/path") == "example.co.uk"


def test_template_masks_ids_and_drops_scheme_query_and_fragment():
    assert AUDIT.url_template("https://a.example.test/item/123?q=1#x") == (
        "a.example.test/item/#"
    )
