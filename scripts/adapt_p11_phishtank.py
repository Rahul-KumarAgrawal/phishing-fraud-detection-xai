"""Create the P11 phishing adapter from the immutable PhishTank CSV."""
from __future__ import annotations
import hashlib, json, pathlib, tempfile
from datetime import datetime, timezone
import pandas as pd
from prepare_p11_benchmarks import validate_records

ROOT=pathlib.Path(__file__).resolve().parents[1]; RAW=ROOT/"data"/"raw"; OUT=ROOT/"reports"/"p11"
SOURCE=RAW/"verified_online.csv"; TARGET=RAW/"prospective_phishtank_verified.csv"
EXPECTED=("phish_id","url","phish_detail_url","submission_time","verified","verification_time","online","target")
TERMS="https://phishtank.org/terms.php"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def atomic_csv(frame,path):
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",delete=False,dir=path.parent,suffix=".tmp") as f: frame.to_csv(f,index=False); tmp=pathlib.Path(f.name)
    tmp.replace(path)
def main():
    source=pd.read_csv(SOURCE,encoding="utf-8")
    if tuple(source.columns)!=EXPECTED: raise RuntimeError(f"unexpected PhishTank schema: {source.columns.tolist()}")
    if not source.verified.astype(str).str.lower().eq("yes").all() or not source.online.astype(str).str.lower().eq("yes").all(): raise RuntimeError("feed contains non-verified or non-online rows")
    if pd.to_datetime(source.submission_time,utc=True,errors="coerce").isna().any() or pd.to_datetime(source.verification_time,utc=True,errors="coerce").isna().any(): raise RuntimeError("unparseable source timestamp")
    if source.phish_id.duplicated().any(): raise RuntimeError("duplicate phish_id")
    ordered=source.sort_values("phish_id",kind="stable")
    duplicates=ordered[ordered.url.duplicated(keep="first")].copy()
    retained=ordered.drop_duplicates("url",keep="first").copy()
    adapted=pd.DataFrame({"url":retained.url,"label":"phishing","source":"PhishTank verified_online.csv","source_record_id":retained.phish_id.astype(str),"verification_status":"verified","verification_timestamp":retained.verification_time,"source_terms_ref":TERMS})
    errors=validate_records(adapted,"phishing")
    if errors: raise RuntimeError(f"adapter contract failure: {errors}")
    atomic_csv(adapted,TARGET); OUT.mkdir(parents=True,exist_ok=True)
    audit={"status":"complete","source":{"filename":SOURCE.name,"sha256":digest(SOURCE),"bytes":SOURCE.stat().st_size,"terms":TERMS,"filesystem_last_write_time_utc":datetime.fromtimestamp(SOURCE.stat().st_mtime,timezone.utc).isoformat(),"acquisition_time":"unknown; filesystem timestamp is not asserted as acquisition time"},"source_rows":int(len(source)),"source_schema":list(source.columns),"verified_counts":source.verified.value_counts().to_dict(),"online_counts":source.online.value_counts().to_dict(),"timestamp_range":{"submission_time":[str(pd.to_datetime(source.submission_time,utc=True).min()),str(pd.to_datetime(source.submission_time,utc=True).max())],"verification_time":[str(pd.to_datetime(source.verification_time,utc=True).min()),str(pd.to_datetime(source.verification_time,utc=True).max())]},"duplicate_url_rule":"sort phish_id ascending and retain first URL; all source labels are verified phishing","excluded_duplicate_urls":int(len(duplicates)),"retained_rows":int(len(adapted)),"output":{"filename":str(TARGET.relative_to(ROOT)),"sha256":digest(TARGET)},"prospective_eligibility":"not established: feed contains records from 2011 onward and no pre-registered collection cutoff was supplied"}
    (OUT/"p11_phishtank_adapter_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
if __name__=="__main__":main()
