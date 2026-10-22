"""P10 must audit saved P9 aggregates without re-evaluating CompPhish."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports" / "p10" / "p10_p9_integrity_review.json"


def test_p10_p9_integrity_review_is_aggregate_only_and_consistent():
    audit = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert audit["status"] == "complete"
    assert audit["frozen_evaluator_was_not_run_by_this_audit"] is True
    assert audit["class_order"] == [0, 1]
    assert audit["confusion_matrix"] == {"tn": 69, "fp": 4421, "fn": 6, "tp": 1947}
    assert audit["derived_metrics"]["false_positive_rate_legitimate"] == 4421 / 4490
