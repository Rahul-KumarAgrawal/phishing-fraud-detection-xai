"""P7 candidate split and artifact invariants."""

import json
from pathlib import Path

import pandas as pd

from fraud_phishing.models.url_model import _load_artifact, predict_url


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p7" / "p7_split_manifest.json"
DATA = ROOT / "data" / "processed"


def test_p7_manifest_proves_all_prohibited_split_overlap_is_zero():
    manifest = json.loads(REPORT.read_text(encoding="utf-8"))
    assert manifest["isolation"] == {
        "exact_url": 0,
        "normalized_url": 0,
        "registered_domain": 0,
        "template": 0,
    }


def test_p7_candidate_artifacts_load_and_predict_both_classes():
    train = pd.read_csv(DATA / "p7_url_train.csv")
    samples = [train[train.label == 0].url.iloc[0], train[train.label == 1].url.iloc[0]]
    for stem in ("logisticregression", "randomforest"):
        pipe, meta = _load_artifact(
            ROOT / "models" / f"p7_url_{stem}.joblib",
            ROOT / "models" / f"p7_url_{stem}_metadata.json",
        )
        assert list(pipe.classes_) == [0, 1]
        for url in samples:
            result = predict_url(url, pipe, float(meta["threshold"]), meta["feature_names"])
            assert result["label"] == int(result["score"] >= result["threshold"])
            assert result["label_name"] in {"legitimate", "phishing"}
