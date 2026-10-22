"""Verify the actual PhishVN v3.1.0 archive documented by the v4 record."""
from __future__ import annotations
import hashlib, json, pathlib, re, zipfile
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]; ARCHIVE=ROOT/"data"/"raw"/"PhishVN_v3.1.0_open.zip"; OUT=ROOT/"reports"/"p11"
LINE=re.compile(r"^([0-9a-f]{64})  (.+?)  \((\d+) bytes\)$")
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest=z.read("MANIFEST.txt").decode("utf-8",errors="strict")
        checks=[]
        for line in manifest.splitlines():
            match=LINE.match(line)
            if match:
                expected,name,size=match.groups(); data=z.read(name)
                checks.append({"member":name,"expected_sha256":expected,"actual_sha256":sha(data),"expected_bytes":int(size),"actual_bytes":len(data),"valid":sha(data)==expected and len(data)==int(size)})
        if not checks or not all(x["valid"] for x in checks): raise RuntimeError("PhishVN manifest verification failed")
        frame=pd.read_csv(z.open("data/dataset_url.csv"),encoding="utf-8",dtype=str,keep_default_na=False)
    required={"id","url","label","source","collected_at","scraped_at","label_source","split","tier"}
    missing=sorted(required-set(frame.columns))
    if missing: raise RuntimeError(f"PhishVN schema missing {missing}")
    if not set(frame.label.dropna()).issubset({"phishing","benign"}): raise RuntimeError("unexpected PhishVN labels")
    if frame.id.isna().any() or frame.id.duplicated().any() or frame.url.isna().any() or frame.url.duplicated().any(): raise RuntimeError("PhishVN duplicate/missing id or url")
    core=frame[frame.tier.astype(str).str.lower().isin(["gold","silver"]) | frame.label.eq("benign")]
    bronze=frame[frame.tier.astype(str).str.lower().eq("bronze")]
    if not bronze.label.eq("phishing").all() or len(core)+len(bronze)!=len(frame): raise RuntimeError("PhishVN tier separation invalid")
    report={"status":"complete","archive":{"filename":ARCHIVE.name,"sha256":sha(ARCHIVE.read_bytes()),"bytes":ARCHIVE.stat().st_size,"release_provenance":"v3.1.0 archive accepted because official v4 describes v4 as documentation-only with byte-identical data","license":"CC BY 4.0 bundled LICENSE"},"manifest":{"entries":len(checks),"all_valid":True,"checks":checks},"schema":{"columns":frame.columns.tolist(),"labels":frame.label.value_counts().to_dict(),"tiers":frame.tier.value_counts().to_dict(),"splits":frame.split.value_counts().to_dict(),"sources":frame.source.value_counts().to_dict()},"integrity":{"rows":int(len(frame)),"duplicate_ids":0,"duplicate_urls":0,"missing_required_columns":missing,"scraped_at_invalid":int(pd.to_datetime(frame.scraped_at,utc=True,errors="coerce",format="mixed").isna().sum()),"collected_at_blank":int(frame.collected_at.eq("").sum()),"record_identifier":"id"},"separation":{"verified_core_rows":int(len(core)),"verified_core_labels":core.label.value_counts().to_dict(),"bronze_rows":int(len(bronze)),"bronze_labels":bronze.label.value_counts().to_dict(),"rule":"gold/silver phishing plus benign rows form verified core; bronze phishing remains separate"},"limitations":["URLs are scheme-stripped in the dataset documentation; HTTP/HTTPS behavior cannot be evaluated.","This source was examined during development planning and is a secondary benchmark candidate, not a fully blind final test.","No development-overlap exclusion has been applied in this P11.2 intake report."]}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"p11_phishvn_archive_qualification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__":main()
