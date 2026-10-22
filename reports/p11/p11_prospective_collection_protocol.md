# P11 prospective collection protocol v1

## Future PhishTank snapshot

Before download, record a UTC cutoff and collection protocol version in an
immutable run note. Retain retrieval UTC time, source URL, terms snapshot,
original filename, byte size, SHA-256, and maximum `verification_time`. Include
only `verified=yes`, retain `phish_id`, submission/verification timestamps,
`online`, and original URL, then remove post-cutoff rows. A feed containing old
rows is historical diagnostic data until this cutoff is applied. OpenPhish
requires approved academic access and its recorded terms. Do not silently
convert unverified, offline, stale, duplicate, or conflicting records.

## Legitimate evidence acquisition

Start from a versioned, legally reusable organisation registry such as ROR CC0
or an official government registry, but use it only as a seed. Record seed ID,
page URL, official reference, evidence method, observation UTC time, reviewer
provenance, and `confirmed` or `uncertain`. Confirmation requires page-level
official evidence and an observation that the page is under the organisation's
stated web property; ownership or URL existence alone is insufficient. Never
use requests or page content as model inputs.

Compromised hosts, user-generated content, redirects to another host, stale
evidence, feed hits, and conflicts are `uncertain` and excluded. Sample paths
and queries from officially published pages across domains and schemes, record
sampling strata and confidence, and do not manufacture class balance.

Before freeze, deduplicate exact URLs, quarantine label conflicts, and exclude
any exact raw URL, conservative normalized URL, hostname, registered domain, or
project normalized template overlapping PHIUSIL, MURL, LegitPhish, or URL-Phish.
Record all nonexclusive flags and one primary exclusion reason. Do not use
CompPhish in this workflow. Write final CSVs and manifests atomically, hash them
after writing, and expose only manifests during model development.
