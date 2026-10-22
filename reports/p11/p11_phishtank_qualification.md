# P11 PhishTank feed qualification

`data/raw/verified_online.csv` was preserved unchanged. It is UTF-8 without a
BOM, 11,589,341 bytes, and has SHA-256
`eae6fe0459db5f87a4b5109ef1454234e9867a396eb18ea3da3ea4f1b05f6bcd`.
Its filesystem last-write time is recorded in the adapter audit, but acquisition
time is unknown and is not inferred from that timestamp.

The source header is `phish_id,url,phish_detail_url,submission_time,verified,
verification_time,online,target`. It has 60,415 rows, zero missing values,
60,415 `verified=yes`, 60,415 `online=yes`, zero duplicate IDs, three duplicate
raw URLs, and parseable submission/verification timestamps. The verification
range is 2011-02-19 through 2026-10-19. Applicable terms are recorded as
`https://phishtank.org/terms.php`.

The adapter sorts ascending `phish_id`, retains the first of same-label duplicate
URLs, and writes 60,412 contract-valid records to
`data/raw/prospective_phishtank_verified.csv`. It maps the source fields without
inventing a label: `phish_id` becomes `source_record_id`, `yes` becomes the
contract's `verified`, and `verification_time` becomes
`verification_timestamp`. The terms reference is static source provenance.

This is **not a prospective snapshot**: the feed includes historical records
from 2011 and no pre-registered cutoff or acquisition timestamp is available.
The adapted arm also has prohibited development overlap before exclusions:
8,112 exact URLs, 8,113 normalized URLs, 7,965 hostnames, 1,628 registered
domains, and 14,568 templates. It must be filtered with an exclusion ledger
before it can be considered for any freeze.

The P11 benchmark remains unqualified because page-level legitimate evidence is
absent and the required `PhishVN_v4.zip` archive is absent. A local
`PhishVN_v3.1.0_open.zip` was found but is not substituted for the required v4
input without its own verified version/manifest qualification.
