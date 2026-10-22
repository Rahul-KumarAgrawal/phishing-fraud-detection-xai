# P11.1 qualification contracts v1

Raw sources remain immutable. These are canonical **derived-adapter** contracts;
the adapter must record its source column mapping and must not rewrite raw files.

## Prospective phishing

Required fields: `url`, `label=phishing`, `source`, `source_record_id`,
`verification_status=verified`, `verification_timestamp`, and
`source_terms_ref`. Verification timestamps must parse as UTC. Duplicate source
identifiers or raw URLs fail. Unverified, blank, conflicting, or uncertain rows
are excluded rather than coerced.

## Page-level legitimate evidence

Required fields: `url`, `label=legitimate`, `evidence_source`,
`evidence_reference`, `evidence_method`, `evidence_status=confirmed`,
`observation_timestamp`, and `review_provenance`. A domain registry record is
only a seed: page-level evidence must be recorded. `uncertain` is excluded.

## PhishVN v4

The v4 record documents the acquired `PhishVN_v3.1.0_open.zip` as byte-identical
data. Its archive must contain `data/dataset_url.csv` and `MANIFEST.txt`; supplied
splits and the datasheet remain required before assignment of a benchmark role.
Its actual publisher manifest grammar is now verified as SHA-256, member path,
and byte count. The adapter output requires
`url`, `label` (`legitimate`/`phishing`), and `confidence_tier`. Gold/silver
phishing plus legitimate rows are verified core; bronze phishing must be emitted
separately and cannot be relabelled as core.

## Overlap and freeze

Use existing P6 definitions for trimmed exact URL, conservative normalized URL,
hostname, offline-PSL registrable domain, and normalized template. Any hit is
excluded; the deterministic primary reason order is exact, normalized, hostname,
domain, template. Retained rows must have zero flags. Final output and status
files use atomic writes. States are `inputs_missing`, `inputs_present_unqualified`,
`qualification_failed`, and `qualified`; only a completed source adapter,
checksum evidence, provenance, and zero-overlap audit can reach `qualified`.
