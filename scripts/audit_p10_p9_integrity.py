"""Read-only integrity audit of saved P9 aggregate results.

This script never opens the frozen CSV, loads a model, or produces predictions.
It checks arithmetic in the saved P9 report and statically inspects the fixed
evaluator's positive-score and ROC-AUC paths.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
P9 = ROOT / "reports" / "p9" / "p9_frozen_compphish_metrics.json"
EVALUATOR = ROOT / "scripts" / "evaluate_p9_frozen_compphish.py"
OUT = ROOT / "reports" / "p10"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(P9.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (cm[k] for k in ("tn", "fp", "fn", "tp"))
    total, legit, phish = tn + fp + fn + tp, tn + fp, fn + tp
    expected = {
        "accuracy": (tn + tp) / total,
        "balanced_accuracy": ((tn / legit) + (tp / phish)) / 2,
        "false_positive_rate_legitimate": fp / legit,
        "false_negative_rate_phishing": fn / phish,
        "precision": tp / (tp + fp),
        "recall": tp / phish,
        "f1": 2 * tp / (2 * tp + fp + fn),
    }
    for key, value in expected.items():
        if abs(metrics[key] - value) > 1e-12:
            raise RuntimeError(f"P9 saved {key} is inconsistent with its confusion matrix")
    source = EVALUATOR.read_text(encoding="utf-8")
    required = ["scores = model.predict_proba(X)[:, 1]", "roc_auc_score(y, scores)", "average_precision_score(y, scores)", "list(model.classes_) != [0, 1]"]
    missing = [text for text in required if text not in source]
    if missing:
        raise RuntimeError(f"P9 evaluator does not expose expected positive-class/ROC safeguards: {missing}")
    if "pd.read_csv(FROZEN)" in source:
        evaluator_reads_frozen = True
    else:
        evaluator_reads_frozen = False
    output = {"status": "complete", "scope": "aggregate-only P9 arithmetic and evaluator source inspection; no frozen rows, model, or predictions read", "confusion_matrix": cm, "derived_metrics": expected, "class_order": report["input_verification"]["class_order"], "positive_class_score": "predict_proba(X)[:, 1] after class-order assertion [0, 1]", "roc_auc_path": "roc_auc_score(y, scores), where scores is the phishing-class probability", "p9_evaluator_itself_reads_frozen_when_run": evaluator_reads_frozen, "frozen_evaluator_was_not_run_by_this_audit": True}
    (OUT / "p10_p9_integrity_review.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
