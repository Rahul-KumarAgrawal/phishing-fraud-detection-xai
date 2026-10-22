"""Integrity and serving contracts for the final showcase URL model."""
import hashlib
import json
from pathlib import Path

from fraud_phishing.features.url_features import FEATURE_NAMES
from fraud_phishing.models.url_model import _load_artifact, predict_url

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "showcase" / "showcase_url_results.json"
MANIFEST = ROOT / "reports" / "showcase" / "showcase_split_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_showcase_split_has_zero_prohibited_overlap_and_preserves_compphish_copy():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert all(
        value == 0
        for comparison in manifest["overlap"].values()
        for value in comparison.values()
    )
    original = ROOT / manifest["inputs"]["compphish_frozen_path"]
    copied = ROOT / manifest["inputs"]["compphish_development_copy"]
    assert _sha256(original) == manifest["inputs"]["compphish_frozen_sha256"]
    assert _sha256(copied) == manifest["inputs"]["compphish_development_copy_sha256"]
    assert _sha256(original) == _sha256(copied)


def test_integrated_showcase_model_matches_reported_schema_and_prediction_contract():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["integration"]["integrated"] is True
    artifact = ROOT / report["integration"]["served_artifact"]
    metadata = ROOT / report["integration"]["served_metadata"]
    assert _sha256(artifact) == report["integration"]["served_artifact_sha256"]
    pipe, meta = _load_artifact(artifact, metadata)
    names = list(meta["feature_names"])
    assert meta["label_map"] == {"0": "legitimate", "1": "phishing"}
    assert meta["n_features"] == len(names) == int(pipe.n_features_in_)
    assert [FEATURE_NAMES.index(name) for name in names] == sorted(
        FEATURE_NAMES.index(name) for name in names
    )
    result = predict_url("https://example.org/account?item=1", pipe, meta["threshold"], names)
    assert result["label"] == int(result["score"] >= result["threshold"])
    assert result["label_name"] in {"legitimate", "phishing"}


def test_showcase_report_records_candidate_metrics_and_historical_p9_boundary():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert len(report["candidates"]) == 4
    for candidate in report["candidates"].values():
        for partition in ("validation", "test"):
            metrics = candidate[partition]
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert "legitimate_path_query_subgroups" in candidate["test"]
    assert report["scope"]["compphish_role"].startswith("development data")
    assert report["phishvn_diagnostic"]["evaluated"] is False
