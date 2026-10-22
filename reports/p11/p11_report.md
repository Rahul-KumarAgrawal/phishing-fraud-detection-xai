# P11 — Acquisition, Qualification, and Freeze Status

## Status: blocked before row-level qualification

Local inventory found no PhishVN v4 archive and no raw prospective PhishTank,
OpenPhish, ROR-derived, or legitimate-evidence files. No data was downloaded,
no URL destination was visited, no model inference ran, and no benchmark was
created. This is a deliberate fail-closed outcome: missing data is not treated
as an empty qualified dataset.

### A. PhishVN secondary candidate

The official [PhishVN v4 record](https://data.mendeley.com/datasets/b97hxbxtpd/4)
reports CC BY 4.0, a 2,587-phishing/16,410-legitimate verified core, and a
separate 34,119-record bronze phishing stratum. Archive-level integrity, file
inventory, class counts, duplicate/conflict counts, and all five overlap levels
remain unverified because the archive is absent. Manually download the archive
shown on that page as `data/raw/PhishVN_v4.zip` without bypassing access
controls, preserve its original filename and SHA-256, then verify `MANIFEST.txt`
and the supplied dataset/split/datasheet files before qualification. Its
scheme-stripped URLs make it a secondary, locale-specific benchmark only; it
cannot establish HTTP/HTTPS generalization.

### B. Prospective final benchmark

No lawful, page-level labelled prospective pair has been collected. PhishTank
offers verified URL records with `verification_time`; OpenPhish academic use
requires approval; ROR is CC0 metadata but only a legitimate-domain seed, not a
benign-page label. The versioned collection protocol and source registry define
the evidence, retention, uncertain-label, overlap, and atomic-freeze gates.

The future benchmark must exclude overlap at exact URL, normalized URL,
hostname, registered domain, and template against PHIUSIL, MURL, LegitPhish,
and URL-Phish. CompPhish is retired from any new selection workflow. Final
class counts and exclusions cannot be stated until lawful raw inputs arrive.

## Recommendation

Obtain the PhishVN v4 archive manually and obtain approval before using
OpenPhish. Collect a date-bounded PhishTank snapshot plus page-level legitimate
evidence under protocol v1, then run the existing P6 normalization/domain/
template audit before any model work. Do not retrain or evaluate either future
candidate yet.
