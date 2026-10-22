import pandas as pd
from scripts.prepare_p11_benchmarks import validate_records, verify_phishvn_archive, verify_member_hashes, validate_phishvn_rows, overlap_flags
from pathlib import Path

def phishing():
    return pd.DataFrame([{"url":"https://x/a","label":"phishing","source":"PhishTank","source_record_id":"1","verification_status":"verified","verification_timestamp":"2026-01-01T00:00:00Z","source_terms_ref":"terms"}])
def legitimate():
    return pd.DataFrame([{"url":"https://y/a","label":"legitimate","evidence_source":"registry","evidence_reference":"id","evidence_method":"page review","evidence_status":"confirmed","observation_timestamp":"2026-01-01T00:00:00Z","review_provenance":"reviewer"}])
def test_valid_record_contracts():
    assert not validate_records(phishing(), "phishing")
    assert not validate_records(legitimate(), "legitimate")
def test_invalid_labels_timestamps_and_evidence_fail():
    x=phishing(); x.loc[0,"label"]="legitimate"; x.loc[0,"verification_timestamp"]="bad"
    assert validate_records(x,"phishing")
    y=legitimate(); y.loc[0,"evidence_status"]="uncertain"
    assert validate_records(y,"legitimate")
def test_missing_and_duplicate_fields_fail():
    assert validate_records(phishing().drop(columns="source"),"phishing")
    assert "duplicate source_record_id" in validate_records(pd.concat([phishing(),phishing()]),"phishing")
def test_five_overlap_flags_and_primary_reason_are_deterministic():
    flags=overlap_flags(pd.Series(["https://example.com/a1", "https://new.example.net/x"]), pd.Series(["http://example.com/a2"]))
    assert set(["exact_url","normalized_url","hostname","registered_domain","template","primary_exclusion"]).issubset(flags)
    assert flags.loc[0,"hostname"] and flags.loc[0,"registered_domain"]
    assert flags.loc[0,"primary_exclusion"] == "hostname"
def test_invalid_archive_fails(tmp_path):
    bad=tmp_path/"x.zip"; bad.write_text("no")
    assert verify_phishvn_archive(bad) == ["invalid PhishVN zip archive"]
def test_archive_checksums_and_phishvn_tiers(tmp_path):
    import zipfile, hashlib
    archive=tmp_path/"v4.zip"
    with zipfile.ZipFile(archive,"w") as z: z.writestr("dataset_url.csv","url,label\nhttps://a,legitimate\n"); z.writestr("MANIFEST.txt","adapter test")
    digest=hashlib.sha256(b"url,label\nhttps://a,legitimate\n").hexdigest()
    assert not verify_phishvn_archive(archive)
    assert not verify_member_hashes(archive,{"dataset_url.csv":digest})
    assert verify_member_hashes(archive,{"dataset_url.csv":"0"*64})
    valid=pd.DataFrame([{"url":"a","label":"legitimate","confidence_tier":"legitimate"},{"url":"b","label":"phishing","confidence_tier":"gold"},{"url":"c","label":"phishing","confidence_tier":"bronze"}])
    assert not validate_phishvn_rows(valid)
    valid.loc[2,"label"]="legitimate"; assert validate_phishvn_rows(valid)
def test_present_invalid_files_fail_closed(tmp_path, monkeypatch):
    import scripts.prepare_p11_benchmarks as gate
    raw=tmp_path/"raw"; raw.mkdir(); out=tmp_path/"out"
    (raw/"PhishVN_v4.zip").write_text("invalid")
    phishing().drop(columns="source").to_csv(raw/"prospective_phishtank_verified.csv",index=False)
    legitimate().to_csv(raw/"prospective_legitimate_evidence.csv",index=False)
    monkeypatch.setattr(gate,"RAW",raw); monkeypatch.setattr(gate,"OUT",out)
    monkeypatch.setattr(gate,"REQUIRED", {"phishvn_secondary":[raw/"PhishVN_v4.zip"],"prospective_final":[raw/"prospective_phishtank_verified.csv",raw/"prospective_legitimate_evidence.csv"]})
    import pytest
    with pytest.raises(SystemExit): gate.main()
    import json
    assert json.loads((out/"p11_acquisition_status.json").read_text())["status"] == "qualification_failed"
