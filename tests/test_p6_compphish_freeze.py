"""Integrity checks for the frozen, evaluation-only CompPhish candidate."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "p6" / "compphish_external_frozen_manifest.json"


def test_frozen_manifest_records_zero_prohibited_overlap():
    if not MANIFEST.exists():
        raise AssertionError("Run scripts/freeze_compphish_external.py before this test")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["post_filter_overlap"] == {
        "exact_url": 0,
        "normalized_url": 0,
        "hostname": 0,
        "registered_domain": 0,
        "template": 0,
    }


def test_frozen_file_hash_matches_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    path = ROOT / manifest["output"]["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest["output"]["sha256"]
