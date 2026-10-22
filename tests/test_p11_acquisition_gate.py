import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_p11_missing_inputs_are_explicitly_blocked_not_empty():
    status = json.loads((ROOT / "reports" / "p11" / "p11_acquisition_status.json").read_text())
    assert status["status"] == "inputs_missing"
    assert "prospective_final" in status["missing"]
    assert status["missing"]["prospective_final"] == [
        "data\\raw\\prospective_legitimate_evidence.csv"
    ]
