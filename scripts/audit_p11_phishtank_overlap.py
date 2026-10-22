"""Static five-way overlap audit for the adapted PhishTank arm only."""
from __future__ import annotations
import json, pathlib, sys
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
from audit_url_datasets import SOURCES, load_source, _registered_domain
from audit_compphish_overlap import normalized_url, hostname
from fraud_phishing.data.prepare_urls import url_template
from prepare_p11_benchmarks import atomic_json
OUT=ROOT/"reports"/"p11"/"p11_phishtank_development_overlap.json"
def main():
    candidate=pd.read_csv(ROOT/"data"/"raw"/"prospective_phishtank_verified.csv")
    funcs={"exact_url":lambda x:x.strip(),"normalized_url":normalized_url,"hostname":hostname,"registered_domain":_registered_domain,"template":url_template}
    cand={name:set(candidate.url.astype(str).map(fn)) for name,fn in funcs.items()}
    development={name:set() for name in funcs}
    source_rows={}
    for source,spec in SOURCES.items():
        frame=load_source(source,spec,[]); source_rows[source]=int(len(frame))
        for name,fn in funcs.items(): development[name].update(frame.url.astype(str).map(fn))
    results={name:{"candidate_unique":len(cand[name]),"development_unique":len(development[name]),"intersection":len(cand[name]&development[name]),"pct_candidate":round(100*len(cand[name]&development[name])/max(1,len(cand[name])),6)} for name in funcs}
    atomic_json(OUT,{"status":"complete","candidate_rows":int(len(candidate)),"development_sources":list(SOURCES),"development_rows":source_rows,"definitions":{"exact_url":"trimmed raw URL","normalized_url":"P6 conservative normalizer","hostname":"P6 parsed hostname","registered_domain":"P6 offline-PSL extractor","template":"project url_template"},"results":results,"zero_prohibited_overlap":all(x["intersection"]==0 for x in results.values()),"scope":"qualification evidence only; no model load, fit, or prediction"})
if __name__=="__main__":main()
