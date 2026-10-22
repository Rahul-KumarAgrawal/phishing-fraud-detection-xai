"""P8 development-only artifact and protocol invariants."""
import json
from pathlib import Path

from fraud_phishing.models.url_model import _load_artifact, predict_url

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p8" / "p8_experiments.json"


def test_p8_report_records_development_only_boundary_and_grouped_protocol():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["scope"]["development_sources"] == ["PHIUSIL", "MURL"]
    assert report["scope"]["frozen_comp_phish_read"] is False
    assert report["scope"]["new_features"] is False
    assert report["inputs"]["cv_protocol"].startswith("3-fold GroupKFold")
    assert len(report["variants"]) >= 7


def test_p8_candidate_schema_and_probability_threshold_contract():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for candidate in report["p8_candidates"].values():
        pipe, metadata = _load_artifact(ROOT / candidate["artifact"], ROOT / candidate["metadata"])
        result = predict_url("https://example.org/account?item=1", pipe, metadata["threshold"], metadata["feature_names"])
        assert result["label"] == int(result["score"] >= result["threshold"])
        assert list(pipe.classes_) == [0, 1]
