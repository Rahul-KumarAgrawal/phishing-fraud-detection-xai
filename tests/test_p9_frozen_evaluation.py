"""P9 fixed-evaluation contracts and metric calculations."""
import inspect
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_p9_frozen_compphish import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p9" / "p9_frozen_compphish_metrics.json"


def test_p9_metric_calculation_on_small_fixture():
    # scores at threshold 0.5 produce TN=1, FP=1, FN=1, TP=1.
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([.1, .9, .2, .8]), .5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["false_positive_rate_legitimate"] == .5
    assert metrics["false_negative_rate_phishing"] == .5
    assert metrics["predicted_class_counts"] == {"legitimate": 2, "phishing": 2}


def test_p9_report_verifies_immutable_inputs_and_probability_mapping():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    verified = report["input_verification"]
    assert report["status"] == "complete"
    assert verified["frozen_sha256_before"] == verified["frozen_sha256_after"]
    assert verified["class_order"] == [0, 1]
    assert report["operating_point"]["no_retraining_refitting_recalibration_or_tuning"] is True
    assert report["metrics"]["actual_class_counts"] == {"legitimate": 4490, "phishing": 1953}


def test_p9_evaluation_path_has_no_fit_or_threshold_selection():
    source = inspect.getsource(__import__("scripts.evaluate_p9_frozen_compphish", fromlist=["*"]))
    assert ".fit(" not in source
    assert "select_threshold" not in source
