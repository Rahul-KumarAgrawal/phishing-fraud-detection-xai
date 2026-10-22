import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def test_actual_phishvn_archive_manifest_and_core_bronze_split():
    report=json.loads((ROOT/"reports"/"p11"/"p11_phishvn_archive_qualification.json").read_text())
    assert report["status"] == "complete"
    assert report["archive"]["filename"] == "PhishVN_v3.1.0_open.zip"
    assert report["manifest"]["entries"] == 13 and report["manifest"]["all_valid"] is True
    assert report["separation"]["verified_core_labels"] == {"benign":16410,"phishing":2587}
    assert report["separation"]["bronze_labels"] == {"phishing":34119}
