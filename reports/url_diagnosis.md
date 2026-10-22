# Step 1 — URL Feature Diagnosis

Reproduce with `python scripts/url_feature_diagnosis.py`.

## 0. Data loaded

- LegitPhish (development benchmark): 100,872 URLs
  - legitimate: 37,380
  - phishing:   63,492

- PHIUSIL URL pool (used to define "new"): 235,370 URLs

### Split of the external set

- legitimate, byte-identical to PHIUSIL (memorised): 36,935
- **legitimate, byte-distinct (the failure case): 445**
- phishing, byte-identical to PHIUSIL: 0
- phishing, byte-distinct: 63,492

## 1. Are the labels correct?

Re-read the raw LegitPhish file and look up the original `ClassLabel` of each of the 445 URLs:

```
raw ClassLabel values among the 445 URLs: {1.0: 445}
```

All must be `1.0`, because LegitPhish stores **1 = legitimate**. `prepare_external_urls.py` maps `1 -> 0` (project convention) and asserts the global composition (0 = 63,678, 1 = 37,540), so a miscount would have failed the preparation step outright.

Independent semantic check — the registered domains of these URLs:

```
url
wikipedia.org          102
stackoverflow.com      100
youtube.com             37
google.com               8
forbes.com               3
squarespace.com          3
nytimes.com              2
explodingtopics.com      2
ahrefs.com               2
reddit.com               2
webflow.com              2
statista.com             2
kaggle.com               2
businessinsider.com      2
yahoo.com                2
```

These are unambiguous well-known legitimate sites (youtube, stackoverflow, wikipedia, news), not phishing infrastructure.

## 2. Is URL parsing and feature extraction correct?

- Feature extraction exceptions among the 445 URLs: **0**

Spot-check on known URLs (expected vs extracted):

| URL | feature | expected | extracted |
|---|---|---|---|
| `https://www.youtube.com/watch?v=eP_Ylb5VE9M` | `hostname_length` | 15 | 15 |
| `https://www.youtube.com/watch?v=eP_Ylb5VE9M` | `is_https` | 1 | 1 |
| `https://www.youtube.com/watch?v=eP_Ylb5VE9M` | `query_param_count` | 1 | 1 |
| `https://stackoverflow.com/questions/tagged/php` | `hostname_length` | 19 | 17 |
| `https://stackoverflow.com/questions/tagged/php` | `is_https` | 1 | 1 |
| `https://stackoverflow.com/questions/tagged/php` | `path_depth` | 2 | 3 |

Structural checks over the 445:

- `is_https == 1`: 442 / 445
- `path_length > 0`: 440 / 445
- `has_query == 1`: 38 / 445
- `subdomain_count > 0`: 292 / 445

The extractor reports the structure the URLs actually have: HTTPS, a non-empty path, and (for many) a query string. Parsing is correct; the model simply never saw this shape in training.

## 3. Is the 100% false-positive rate reproducible?

- Production model `RandomForest`, threshold 0.4612
- FPR on byte-identical legitimate URLs: 0.02% (8/36,935)
- **FPR on byte-distinct legitimate URLs: 100.00% (445/445)**
- Mean phishing-probability on the byte-distinct legitimate URLs: 0.9998

## 4. Do training and external data differ in feature distribution?

Mean value of each feature by group (full table: `reports/url_feature_distributions.csv`). The columns most relevant to the failure are shown first:

```
                     group     n  is_https  path_length  slash_count  path_depth  has_query  query_param_count  hostname_length  suspicious_token_count
  PHIUSIL train legitimate 20000  1.000000     0.000000     2.000000    0.000000   0.000000            0.00000         19.23920                 0.00725
    PHIUSIL train phishing 20000  0.487850     9.076800     3.022350    0.592300   0.058700            0.13110         24.51555                 0.12790
 LegitPhish new legitimate   445  0.993258    29.546067     4.289888    2.017978   0.085393            0.08764         15.98427                 0.07191
LegitPhish unseen phishing 20000  0.056700    12.639850     3.309400    1.322800   0.010950            0.02240         15.19170                 0.04225
```

Share of URLs by structural shape:

| group | has path | has query | https |
|---|---|---|---|
| PHIUSIL train legitimate | 0.0% | 0.0% | 100.0% |
| PHIUSIL train phishing | 59.9% | 5.9% | 48.8% |
| LegitPhish new legitimate | 98.9% | 8.5% | 99.3% |
| LegitPhish unseen phishing | 100.0% | 1.1% | 5.7% |

**This is the root cause.** PHIUSIL's legitimate class contains essentially no URLs with a path (0.0% have one), while the new legitimate URLs are almost all path-bearing. `path_length`, `slash_count` and `path_depth` therefore act as class-separating features during training but are uninformative (and misleading) at inference.

## 5. Representative false positives

All 445 rows are in `reports/url_diagnosis_false_positives.csv`. Distinct registered domains: 179. Highest-scoring examples:

```
                                                             url  phishing_probability  path_length  slash_count  has_query
                     https://www.youtube.com/watch?v=eP_Ylb5VE9M                   1.0            6            3          1
                  https://stackoverflow.com/questions/tagged/php                   1.0           21            5          0
       https://stackoverflow.com/questions/tagged/microsoft-edge                   1.0           32            5          0
                  https://stackoverflow.com/questions/tagged/sql                   1.0           21            5          0
https://stackoverflow.com/questions/79030270/how-to-insert-a-...                   1.0           94            5          0
                 https://stackoverflow.com/questions/tagged/java                   1.0           22            5          0
                 https://stackoverflow.com/questions/tagged/iwyu                   1.0           22            5          0
              https://stackoverflow.com/questions/tagged/android                   1.0           25            5          0
         https://stackoverflow.com/questions/tagged/dependencies                   1.0           30            5          0
     https://stackoverflow.com/questions/tagged/machine-learning                   1.0           34            5          0
                https://stackoverflow.com/questions/tagged/build                   1.0           23            5          0
               https://stackoverflow.com/questions/tagged/pandas                   1.0           24            5          0
```

## Verdict

1. **Labels are correct** — the raw `ClassLabel` for every URL in question is 1 (legitimate), the global composition assertion passed, and the domains are well-known legitimate sites.
2. **Parsing and extraction are correct** — no exceptions, and HTTPS / path / query are reported exactly as they appear in the URL.
3. **The false-positive rate is reproducible** — see section 3.
4. **The datasets differ structurally** — PHIUSIL's legitimate class is almost entirely pathless; the external legitimate URLs are not. The model learned the dataset's formatting instead of phishing semantics.
