"""Fail-closed P11 acquisition gate; it never creates an empty benchmark."""
from __future__ import annotations
import hashlib, json, pathlib, tempfile
import zipfile
import pandas as pd
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "p11"
REQUIRED = {
    # Official v4 documents this v3.1.0 archive as byte-identical data.
    "phishvn_secondary": [RAW / "PhishVN_v3.1.0_open.zip"],
    "prospective_final": [RAW / "prospective_phishtank_verified.csv", RAW / "prospective_legitimate_evidence.csv"],
}
# Canonical *derived adapter* contracts. Raw publisher files are preserved and
# source-specific mappings must be documented before these schemas are emitted.
PHISHING_COLUMNS = ("url", "label", "source", "source_record_id", "verification_status", "verification_timestamp", "source_terms_ref")
LEGITIMATE_COLUMNS = ("url", "label", "evidence_source", "evidence_reference", "evidence_method", "evidence_status", "observation_timestamp", "review_provenance")
PHISHVN_REQUIRED_MEMBERS = {"dataset_url.csv", "MANIFEST.txt"}

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def display_path(path: pathlib.Path) -> str:
    try: return str(path.relative_to(ROOT))
    except ValueError: return str(path)

def atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as f:
        json.dump(payload, f, indent=2); temp = pathlib.Path(f.name)
    temp.replace(path)

def validate_records(frame: pd.DataFrame, kind: str) -> list[str]:
    required = PHISHING_COLUMNS if kind == "phishing" else LEGITIMATE_COLUMNS
    errors = [f"missing column {col}" for col in required if col not in frame]
    if errors: return errors
    for col in required:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any(): errors.append(f"blank {col}")
    expected = "phishing" if kind == "phishing" else "legitimate"
    if not frame.label.eq(expected).all(): errors.append(f"label must be {expected}")
    timestamp = "verification_timestamp" if kind == "phishing" else "observation_timestamp"
    if pd.to_datetime(frame[timestamp], utc=True, errors="coerce").isna().any(): errors.append(f"invalid {timestamp}")
    if frame.url.duplicated().any(): errors.append("duplicate raw url")
    if kind == "phishing":
        if not frame.verification_status.str.lower().eq("verified").all(): errors.append("phishing must be verified")
        if frame.source_record_id.duplicated().any(): errors.append("duplicate source_record_id")
    elif not frame.evidence_status.str.lower().eq("confirmed").all(): errors.append("legitimate evidence must be confirmed; uncertain is excluded")
    return errors

def verify_phishvn_archive(path: pathlib.Path) -> list[str]:
    """Check only documented members; checksum grammar awaits real MANIFEST.txt."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = {pathlib.PurePosixPath(n).name for n in archive.namelist()}
            return [f"archive missing {name}" for name in sorted(PHISHVN_REQUIRED_MEMBERS - names)]
    except zipfile.BadZipFile:
        return ["invalid PhishVN zip archive"]

def verify_member_hashes(path: pathlib.Path, expected: dict[str, str]) -> list[str]:
    """Verify adapter-supplied publisher manifest entries without guessing syntax."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            by_base = {pathlib.PurePosixPath(name).name: name for name in archive.namelist()}
            for member, digest in expected.items():
                if member not in by_base: errors.append(f"manifest member missing {member}"); continue
                actual = hashlib.sha256(archive.read(by_base[member])).hexdigest()
                if actual.lower() != digest.lower(): errors.append(f"checksum mismatch {member}")
    except zipfile.BadZipFile:
        errors.append("invalid PhishVN zip archive")
    return errors

def validate_phishvn_rows(frame: pd.DataFrame) -> list[str]:
    """Validate a documented derived adapter output, retaining tiers separately."""
    required = ("url", "label", "confidence_tier")
    errors = [f"missing PhishVN adapter column {col}" for col in required if col not in frame]
    if errors: return errors
    if frame.url.isna().any() or frame.url.astype(str).str.strip().eq("").any(): errors.append("blank PhishVN url")
    if frame.label.isna().any(): errors.append("blank PhishVN label")
    allowed = {"legitimate", "phishing"}
    if not set(frame.label.astype(str)).issubset(allowed): errors.append("unsupported PhishVN label")
    tiers = set(frame.confidence_tier.astype(str).str.lower())
    if not tiers.issubset({"gold", "silver", "bronze", "legitimate"}): errors.append("unsupported PhishVN confidence tier")
    if frame.url.duplicated().any(): errors.append("duplicate PhishVN raw url")
    # A bronze positive cannot be relabelled into the verified core.
    if ((frame.confidence_tier.astype(str).str.lower() == "bronze") & (frame.label.astype(str) != "phishing")).any(): errors.append("bronze tier must remain phishing and separate")
    return errors

def overlap_flags(candidate_urls: pd.Series, development_urls: pd.Series) -> pd.DataFrame:
    """Five P6-defined overlap checks; retained rows require every flag false."""
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    from audit_compphish_overlap import normalized_url
    from audit_url_datasets import _registered_domain
    from fraud_phishing.data.prepare_urls import url_template
    dev = development_urls.astype(str)
    keys = {"exact_url": set(dev.str.strip()), "normalized_url": set(dev.map(normalized_url)), "hostname": set(dev.map(lambda u: normalized_url(u).split("//",1)[-1].split("/",1)[0].split(":",1)[0])), "registered_domain": set(dev.map(_registered_domain)), "template": set(dev.map(url_template))}
    raw = candidate_urls.astype(str)
    values = {"exact_url": raw.str.strip(), "normalized_url": raw.map(normalized_url), "hostname": raw.map(lambda u: normalized_url(u).split("//",1)[-1].split("/",1)[0].split(":",1)[0]), "registered_domain": raw.map(_registered_domain), "template": raw.map(url_template)}
    out = pd.DataFrame({name: values[name].isin(keys[name]) for name in keys})
    order = list(keys)
    out["primary_exclusion"] = out.apply(lambda row: next((name for name in order if row[name]), "retained"), axis=1)
    return out

def main() -> None:
    missing = {name: [str(p.relative_to(ROOT)) for p in paths if not p.is_file()] for name, paths in REQUIRED.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        atomic_json(OUT / "p11_acquisition_status.json", {"status": "inputs_missing", "missing": missing, "rule": "No output CSV, manifest, or zero-row benchmark is created when required raw evidence is missing."})
        raise SystemExit("P11 acquisition blocked: " + json.dumps(missing))
    errors = verify_phishvn_archive(RAW / "PhishVN_v4.zip")
    try:
        errors += validate_records(pd.read_csv(RAW / "prospective_phishtank_verified.csv"), "phishing")
        errors += validate_records(pd.read_csv(RAW / "prospective_legitimate_evidence.csv"), "legitimate")
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        errors.append(f"CSV read failure: {exc}")
    payload = {"status": "qualification_failed" if errors else "inputs_present_unqualified", "inputs": {name: [{"path": display_path(p), "sha256": sha256(p)} for p in paths] for name, paths in REQUIRED.items()}, "errors": errors, "next_gate": "source-specific PhishVN adapter must verify publisher manifest/checksums, separate core/bronze, and run five-way overlap audit before qualified"}
    atomic_json(OUT / "p11_acquisition_status.json", payload)
    raise SystemExit("Inputs are present but remain unqualified; run the documented row-level audit before freezing.")

if __name__ == "__main__": main()
