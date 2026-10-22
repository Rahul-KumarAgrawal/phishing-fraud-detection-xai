# P11.2 intake reconciliation

The original `PhishVN_v3.1.0_open.zip` remains unchanged. Its SHA-256 is
`308351e9d0c0fca13a81f2f63524a7dee08ccba6c775848d125b3da5525d0ab5` and its
size is 5,813,309 bytes. The official v4 record describes v4 as a
documentation-only update with byte-identical data, so this filename is now the
accepted P11 intake archive while the report records that v4 provenance. All 13
internal `MANIFEST.txt` member SHA-256 and byte-count entries verified.

`dataset_url.csv` has 53,116 unique IDs and URLs: 36,706 phishing and 16,410
benign. Gold/silver phishing plus benign data forms the 18,997-row verified core
(2,587 phishing and 16,410 benign). The 34,119 bronze phishing rows remain a
distinct tier. The bundled license is CC BY 4.0. The schema documents that URL
normalization strips schemes, so the archive cannot support HTTP/HTTPS feature
claims. PhishVN remains a secondary, previously examined candidate; no overlap
filter or model evaluation was run in P11.2.

The currently adapted PhishTank feed is historical/diagnostic only. It has no
pre-registered cutoff and includes records from 2011 onward. The updated
prospective protocol requires a future feed snapshot to record cutoff,
retrieval time, source terms, original hash, source IDs, verification times, and
post-cutoff filtering before all five development-overlap exclusions.

The final prospective benchmark is still blocked by missing page-level
legitimate evidence. ROR or a government registry may seed candidates, but each
URL needs a recorded official page-level reference, method, observation time,
review provenance, confidence, and confirmed/uncertain decision. Domain
ownership and URL existence alone are insufficient.
