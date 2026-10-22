"""
download_data.py
================
Downloads the two raw datasets required for the baseline.

  A) SMS/Smishing dataset from Mendeley Data
     Source: https://data.mendeley.com/datasets/f45bkkt8pr/1
     Expected: ~5,971 messages (ham, spam, smishing)

  B) PHIUSIL Phishing URL dataset from UCI ML Repository
     Source: https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
     Expected: ~235,795 records

Usage (from repo root):
    python src/fraud_phishing/data/download_data.py
"""

import hashlib
import os
import sys
import pathlib
import textwrap

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Dataset A – Mendeley SMS/Smishing
# ─────────────────────────────────────────────────────────
MENDELEY_DATASET_PAGE = "https://data.mendeley.com/datasets/f45bkkt8pr/1"
MENDELEY_API_URL = "https://data.mendeley.com/public-api/datasets/f45bkkt8pr"
MENDELEY_EXPECTED_FILE = RAW_DIR / "SMS_dataset.csv"

# The archive and its digest were resolved from the Mendeley public API on
# 2026-10-18 and are pinned here so the download still works when the API is
# behind bot protection. The SHA-256 is always verified before use.
#   Dataset_5971.zip -> Dataset_5971.csv (LABEL, TEXT, URL, EMAIL, PHONE)
MENDELEY_ARCHIVE_NAME = "Dataset_5971.zip"
MENDELEY_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded"
)
MENDELEY_ARCHIVE_SHA256 = (
    "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3"
)

# Composition documented on the dataset card; verified after download.
MENDELEY_EXPECTED_TOTAL = 5_971
MENDELEY_EXPECTED_LABEL_COUNTS = {"ham": 4_844, "spam": 489, "smishing": 638}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _http_get(url: str, timeout: int = 120) -> bytes:
    """
    Fetch a URL and return raw bytes.

    Mendeley sits behind Cloudflare, which rejects the python-requests TLS
    fingerprint from some networks (HTTP 403, "Just a moment..."). We therefore
    try requests first and fall back to the stdlib urllib client.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    errors: list[str] = []

    try:
        import requests  # noqa: PLC0415

        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and resp.content:
            return resp.content
        errors.append(f"requests -> HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"requests -> {type(exc).__name__}: {exc}")

    try:
        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if data:
            return data
        errors.append("urllib -> empty response")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"urllib -> {type(exc).__name__}: {exc}")

    raise RuntimeError("; ".join(errors))


def _resolve_mendeley_archive() -> tuple[str, str]:
    """Return (download_url, sha256) for the archive, preferring the live API."""
    try:
        import json  # noqa: PLC0415

        payload = json.loads(_http_get(MENDELEY_API_URL, timeout=60).decode("utf-8"))
        for entry in payload.get("files", []):
            if entry.get("filename") == MENDELEY_ARCHIVE_NAME:
                details = entry.get("content_details", {})
                url, sha = details.get("download_url"), details.get("sha256_hash")
                if url and sha:
                    print(f"[A] Resolved {MENDELEY_ARCHIVE_NAME} via the Mendeley API.")
                    return url, sha
    except Exception as exc:  # noqa: BLE001
        print(
            f"[A] Mendeley API unavailable ({type(exc).__name__}); "
            "using the pinned archive URL."
        )
    return MENDELEY_ARCHIVE_URL, MENDELEY_ARCHIVE_SHA256


def _extract_message_csv(payload: bytes, source_name: str):
    """Extract the message CSV from a zip archive or a raw CSV payload."""
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            candidates = [
                n
                for n in archive.namelist()
                if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
            ]
            if not candidates:
                raise RuntimeError(
                    f"No CSV found inside {source_name}: {archive.namelist()}"
                )
            name = max(candidates, key=lambda n: archive.getinfo(n).file_size)
            raw = archive.read(name)
            print(f"[A] Extracted {name} from {source_name}.")
    else:
        raw = payload

    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return __import__("pandas").read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not parse the message CSV: {last_error}")


def _verify_mendeley(df) -> tuple[str, str]:
    """Fail loudly unless the dataset matches its documented composition."""
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    text_col = next((c for c in df.columns if c.lower() == "text"), None)
    if label_col is None or text_col is None:
        raise ValueError(f"Expected LABEL and TEXT columns, found {list(df.columns)}")

    if len(df) != MENDELEY_EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {MENDELEY_EXPECTED_TOTAL:,} rows, found {len(df):,}."
        )

    observed = {
        str(k): int(v)
        for k, v in df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .items()
    }
    if observed != MENDELEY_EXPECTED_LABEL_COUNTS:
        raise ValueError(
            "Raw label composition does not match the dataset card "
            f"(expected {MENDELEY_EXPECTED_LABEL_COUNTS}, found {observed})."
        )

    print(
        f"[A] Verified composition: ham={observed['ham']:,} "
        f"smishing={observed['smishing']:,} spam={observed['spam']:,} "
        f"(total {len(df):,})"
    )
    return label_col, text_col


def download_mendeley(force: bool = False) -> bool:
    """
    Download the real Mendeley SMS/Smishing dataset (Dataset A).

    The original implementation hard-coded a non-existent file UUID and relied
    on ``requests`` (which Cloudflare rejects), so it always fell through to the
    manual instructions and left the synthetic placeholder in place. This
    version resolves the archive from the public API, verifies its SHA-256,
    extracts the message CSV, and validates the documented composition.

    Returns True if data/raw/SMS_dataset.csv exists and validates.
    """
    if MENDELEY_EXPECTED_FILE.exists() and not force:
        print(f"[A] SMS dataset already present: {MENDELEY_EXPECTED_FILE}")
        return True

    import hashlib  # noqa: PLC0415

    url, expected_sha = _resolve_mendeley_archive()
    print(f"[A] Downloading {MENDELEY_ARCHIVE_NAME} ...")
    try:
        payload = _http_get(url)
    except Exception as exc:  # noqa: BLE001
        print(f"[A] Download failed: {exc}")
        _print_mendeley_manual_instructions()
        return False

    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha:
        print(
            f"[A] SHA-256 mismatch!\n     expected {expected_sha}\n     got      {digest}"
        )
        _print_mendeley_manual_instructions()
        return False
    print(f"[A] SHA-256 verified ({len(payload):,} bytes).")

    try:
        df = _extract_message_csv(payload, MENDELEY_ARCHIVE_NAME)
        label_col, text_col = _verify_mendeley(df)
    except Exception as exc:  # noqa: BLE001
        print(f"[A] Dataset validation failed: {exc}")
        _print_mendeley_manual_instructions()
        return False

    out = df[[label_col, text_col]].rename(columns={label_col: "LABEL", text_col: "TEXT"})
    out.to_csv(MENDELEY_EXPECTED_FILE, index=False, encoding="utf-8")
    print(f"[A] Wrote {len(out):,} rows -> {MENDELEY_EXPECTED_FILE}")
    print(
        "[A] NOTE: this file replaces the synthetic placeholder. Re-run\n"
        "         src/fraud_phishing/data/prepare_messages.py and\n"
        "         scripts/train_message_model.py to retrain on real data."
    )
    return True


def _print_mendeley_manual_instructions() -> None:
    print(
        textwrap.dedent(
            """
        ====================================================================
        MANUAL DOWNLOAD REQUIRED -- SMS/Smishing Dataset (Dataset A)
        ====================================================================
          1. Open a browser and go to:
             https://data.mendeley.com/datasets/f45bkkt8pr/1
          2. Download "Dataset_5971.zip" and extract Dataset_5971.csv.
          3. Save it as:
             data/raw/SMS_dataset.csv

          Expected columns: LABEL, TEXT (plus URL, EMAIL, PHONE)
          Expected rows: 5,971 (4,844 ham / 638 smishing / 489 spam)
          SHA-256 of Dataset_5971.zip:
            9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3
          License: CC BY 4.0
        ====================================================================
        """
        )
    )


# ─────────────────────────────────────────────────────────
# Dataset B – UCI PHIUSIL Phishing URL Dataset
# ─────────────────────────────────────────────────────────
PHIUSIL_EXPECTED_FILE = RAW_DIR / "PHIUSIL_urls.csv"


def download_phiusil() -> bool:
    """
    Download the PHIUSIL dataset via ucimlrepo (official interface).
    Saves only the URL string and target label columns.

    Returns True if the file is present after the attempt.
    """
    if PHIUSIL_EXPECTED_FILE.exists():
        print(f"[B] URL dataset already present: {PHIUSIL_EXPECTED_FILE}")
        return True

    print("[B] Attempting to fetch PHIUSIL URL dataset via ucimlrepo ...")
    try:
        import io, sys as _sys
        # ucimlrepo may print Unicode chars that crash on Windows cp1252
        # Temporarily redirect stdout to a safe wrapper
        old_stdout = _sys.stdout
        _sys.stdout = io.TextIOWrapper(
            _sys.stdout.buffer, encoding='utf-8', errors='replace'
        ) if hasattr(_sys.stdout, 'buffer') else _sys.stdout
        try:
            from ucimlrepo import fetch_ucirepo  # noqa: PLC0415
            dataset = fetch_ucirepo(id=967)
        finally:
            _sys.stdout = old_stdout
        X = dataset.data.features
        y = dataset.data.targets

        import pandas as pd  # noqa: PLC0415

        # The dataset has a 'URL' column and a target label
        # Inspect actual columns first
        print(f"  Feature columns: {list(X.columns)}")
        print(f"  Target columns:  {list(y.columns)}")

        # Identify the raw URL column (case-insensitive search)
        url_col = _find_column(X, ["url", "URL", "Url"])
        if url_col is None:
            print("  [WARN] No URL column found in features; checking targets …")
            url_col = _find_column(y, ["url", "URL", "Url"])
            if url_col is not None:
                url_series = y[url_col]
            else:
                print("  [ERROR] Cannot locate URL column – saving full features for inspection.")
                X.to_csv(RAW_DIR / "PHIUSIL_features_full.csv", index=False)
                y.to_csv(RAW_DIR / "PHIUSIL_targets_full.csv", index=False)
                _print_phiusil_manual_instructions()
                return False
        else:
            url_series = X[url_col]

        # Identify label column
        label_col = y.columns[0]
        label_series = y[label_col]

        df_out = pd.DataFrame({"url": url_series.values, "label_raw": label_series.values})
        df_out.to_csv(PHIUSIL_EXPECTED_FILE, index=False)
        print(
            f"[B] Saved {len(df_out):,} rows → {PHIUSIL_EXPECTED_FILE}"
        )
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"[B] ucimlrepo fetch failed: {exc}")
        _print_phiusil_manual_instructions()
        return False


def _find_column(df, candidates):
    """Return the first matching column name (case-insensitive)."""
    df_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_lower:
            return df_lower[cand.lower()]
    return None


def _print_phiusil_manual_instructions() -> None:
    print(
        textwrap.dedent(
            """
        ====================================================================
        MANUAL DOWNLOAD REQUIRED -- PHIUSIL URL Dataset (Dataset B)
        ====================================================================
          1. Open a browser and go to:
             https://archive.ics.uci.edu/dataset/967
          2. Download the dataset zip.
          3. Extract and locate the main CSV (usually
             PHIUSIL_Phishing_URL_Dataset.csv or similar).
          4. Place it at:  data/raw/PHIUSIL_urls.csv

          Required columns: URL (string), label (0/1 or -1/1)
          Expected rows: ~235,795
          License: CC BY 4.0
        ====================================================================
        """
        )
    )


# ─────────────────────────────────────────────────────────
# Dataset C – LegitPhish (independent external URL evaluation set)
# ─────────────────────────────────────────────────────────
# This dataset is deliberately NOT used for training. It exists so that the URL
# model can be scored on URLs from a *different* source than PHIUSIL, which is
# the only way to tell whether the headline metrics are real or an artefact of
# one dataset's URL formatting.
LEGITPHISH_DATASET_PAGE = "https://data.mendeley.com/datasets/hx4m73v2sf/2"
LEGITPHISH_API_URL = "https://data.mendeley.com/public-api/datasets/hx4m73v2sf"
LEGITPHISH_EXPECTED_FILE = RAW_DIR / "LegitPhish_urls.csv"

# Resolved from the Mendeley public API on 2026-10-18 and pinned for
# reproducibility. SHA-256 is always verified before use.
LEGITPHISH_ARCHIVE_NAME = "url_features_extracted1.csv"
LEGITPHISH_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/hx4m73v2sf/files/"
    "0ed51e4d-d160-4047-b650-3f0801eea4aa/file_downloaded"
)
LEGITPHISH_ARCHIVE_SHA256 = (
    "8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd"
)

# Documented composition. NOTE the inverted convention, identical to PHIUSIL:
#     ClassLabel 0 = PHISHING, ClassLabel 1 = LEGITIMATE
# The dataset card states: "Target label: 1 = Legitimate, 0 = Phishing", and the
# observed top domains confirm it (label 1 -> wikipedia.org, senate.gov, af.mil).
# One row has a missing label, so the documented counts sum to 101,218.
LEGITPHISH_EXPECTED_TOTAL = 101_219
LEGITPHISH_EXPECTED_LABEL_COUNTS = {1: 37_540, 0: 63_678}  # 1=legit, 0=phishing
LEGITPHISH_RAW_LABEL_MEANING = {1: "legitimate", 0: "phishing"}


def download_legitphish(force: bool = False) -> bool:
    """
    Download the LegitPhish dataset (Dataset C) used for external evaluation.

    Returns True if data/raw/LegitPhish_urls.csv exists and validates.
    """
    if LEGITPHISH_EXPECTED_FILE.exists() and not force:
        print(f"[C] LegitPhish already present: {LEGITPHISH_EXPECTED_FILE}")
        return True

    import hashlib  # noqa: PLC0415

    print(f"[C] Downloading {LEGITPHISH_ARCHIVE_NAME} ...")
    try:
        payload = _http_get(LEGITPHISH_ARCHIVE_URL)
    except Exception as exc:  # noqa: BLE001
        print(f"[C] Download failed: {exc}")
        _print_legitphish_manual_instructions()
        return False

    digest = hashlib.sha256(payload).hexdigest()
    if digest != LEGITPHISH_ARCHIVE_SHA256:
        print(
            f"[C] SHA-256 mismatch!\n     expected {LEGITPHISH_ARCHIVE_SHA256}"
            f"\n     got      {digest}"
        )
        _print_legitphish_manual_instructions()
        return False
    print(f"[C] SHA-256 verified ({len(payload):,} bytes).")

    import pandas as pd  # noqa: PLC0415

    try:
        import io  # noqa: PLC0415

        df = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001
        print(f"[C] Could not parse the CSV: {exc}")
        _print_legitphish_manual_instructions()
        return False

    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    label_col = next((c for c in df.columns if c.lower() == "classlabel"), None)
    if url_col is None or label_col is None:
        print(f"[C] Expected URL and ClassLabel columns, found {list(df.columns)}")
        _print_legitphish_manual_instructions()
        return False

    if len(df) != LEGITPHISH_EXPECTED_TOTAL:
        print(f"[C] Expected {LEGITPHISH_EXPECTED_TOTAL:,} rows, found {len(df):,}.")
        return False

    labels = df[label_col].dropna().astype(int)
    observed = {int(k): int(v) for k, v in labels.value_counts().items()}
    if any(
        observed.get(raw) != expected
        for raw, expected in LEGITPHISH_EXPECTED_LABEL_COUNTS.items()
    ):
        print(
            "[C] Label composition does not match the dataset card "
            f"(expected {LEGITPHISH_EXPECTED_LABEL_COUNTS}, found {observed})."
        )
        return False

    missing = int(df[label_col].isna().sum())
    print(
        f"[C] Verified composition: phishing(0)={observed[0]:,} "
        f"legitimate(1)={observed[1]:,} unlabelled={missing}"
    )

    # Keep only the raw URL and the label: the other 16 columns are LegitPhish's
    # own engineered features, which we must NOT feed to our model (the whole
    # point is to recompute our own feature schema).
    out = df[[url_col, label_col]].rename(columns={url_col: "URL", label_col: "ClassLabel"})
    out = out.dropna(subset=["ClassLabel"]).reset_index(drop=True)
    out.to_csv(LEGITPHISH_EXPECTED_FILE, index=False, encoding="utf-8")
    print(f"[C] Wrote {len(out):,} rows (URL + label only) -> {LEGITPHISH_EXPECTED_FILE}")
    print(
        "[C] WARNING: LegitPhish uses ClassLabel 0 = phishing, 1 = legitimate "
        "(same inverted convention as PHIUSIL).\n"
        "         It is an EXTERNAL EVALUATION set and must not be used for "
        "training."
    )
    return True


def _print_legitphish_manual_instructions() -> None:
    print(
        textwrap.dedent(
            """
        ====================================================================
        MANUAL DOWNLOAD REQUIRED -- LegitPhish URL Dataset (Dataset C)
        ====================================================================
          1. Open a browser and go to:
             https://data.mendeley.com/datasets/hx4m73v2sf/2
          2. Download "url_features_extracted1.csv".
          3. Save it as:
             data/raw/LegitPhish_urls.csv

          SHA-256 of url_features_extracted1.csv:
            8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd
          Expected rows: 101,219 (63,678 phishing / 37,540 legitimate)
          Labels: ClassLabel 0 = phishing, 1 = legitimate
          License: CC BY 4.0
        ====================================================================
        """
        )
    )


# ─────────────────────────────────────────────────────────
# Dataset D – MURL (training augmentation only)
# ─────────────────────────────────────────────────────────
# MURL is a large multi-class corpus whose source differs from PHIUSIL: 0 URLs
# in it are byte-identical to PHIUSIL, and 86% of its URLs carry a path, which is
# exactly the shape PHIUSIL's legitimate class lacks. It is used ONLY to augment
# training data; no evaluation split is ever carved out of it.
MURL_DATASET_PAGE = "https://data.mendeley.com/datasets/5fk24y4xmg/2"
MURL_EXPECTED_FILE = RAW_DIR / "MURL_urls.csv"
MURL_ARCHIVE_NAME = "Malicious URL Detection.csv"
MURL_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/5fk24y4xmg/files/"
    "3d95c20f-afdf-46d3-b85b-bdf049ca57fd/file_downloaded"
)
MURL_ARCHIVE_SHA256 = (
    "6949ebaf7d9f2b315e899982d99e51a1ffbb4cb9bb433cb324e60c143aa7ceef"
)
MURL_EXPECTED_TOTAL = 783_705
MURL_KEEP_TYPES = {"benign": 0, "legitimate": 0, "phishing": 1}

# ─────────────────────────────────────────────────────────
# Dataset E – URL-Phish (final, untouched external evaluation only)
# ─────────────────────────────────────────────────────────
# URL-Phish is the dataset reserved for the *final* generalisation claim. Its
# benign URLs come from the Research Organization Registry (universities and
# research organisations) and its phishing URLs come from PhishTank -- two
# collection pipelines that neither PHIUSIL nor LegitPhish used. 89% of its URLs
# carry a path, and its HTTPS rate is INVERTED relative to PHIUSIL (benign 35%,
# phishing 91%), so it cannot be passed by the HTTPS shortcut.
#
# NOTE the label convention is the *intuitive* one (0 = benign, 1 = phishing),
# i.e. the OPPOSITE of PHIUSIL and LegitPhish. It is asserted on load.
URLPHISH_DATASET_PAGE = "https://data.mendeley.com/datasets/65z9twcx3r/2"
URLPHISH_EXPECTED_FILE = RAW_DIR / "URLPhish_urls.csv"
URLPHISH_ARCHIVE_NAME = "Dataset.csv"
URLPHISH_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/65z9twcx3r/files/"
    "0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded"
)
URLPHISH_ARCHIVE_SHA256 = (
    "d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1"
)
URLPHISH_EXPECTED_TOTAL = 116_600
URLPHISH_EXPECTED_LABEL_COUNTS = {0: 100_000, 1: 16_600}  # 0=benign, 1=phishing


def _fetch_verified(url: str, expected_sha256: str, tag: str) -> bytes | None:
    """Download a URL and verify its SHA-256, or return None on failure."""
    try:
        payload = _http_get(url)
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] Download failed: {exc}")
        return None
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        print(
            f"[{tag}] SHA-256 mismatch!\n     expected {expected_sha256}"
            f"\n     got      {digest}"
        )
        return None
    print(f"[{tag}] SHA-256 verified ({len(payload):,} bytes).")
    return payload


def download_murl(force: bool = False) -> bool:
    """
    Download MURL (Dataset D) and keep only the class column used for training.

    Returns True if data/raw/MURL_urls.csv exists and validates.
    """
    if MURL_EXPECTED_FILE.exists() and not force:
        print(f"[D] MURL already present: {MURL_EXPECTED_FILE}")
        return True

    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    print(f"[D] Downloading {MURL_ARCHIVE_NAME} ...")
    payload = _fetch_verified(MURL_ARCHIVE_URL, MURL_ARCHIVE_SHA256, "D")
    if payload is None:
        print(f"[D] Manual download: {MURL_DATASET_PAGE}")
        return False

    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    if len(df) != MURL_EXPECTED_TOTAL:
        print(f"[D] Expected {MURL_EXPECTED_TOTAL:,} rows, found {len(df):,}.")
        return False
    if not {"url", "type"}.issubset(df.columns):
        print(f"[D] Expected 'url' and 'type' columns, found {list(df.columns)}")
        return False

    counts = df["type"].value_counts().to_dict()
    kept = df[df["type"].isin(MURL_KEEP_TYPES)].copy()
    kept["label"] = kept["type"].map(MURL_KEEP_TYPES).astype(int)
    kept = kept[["url", "label"]]
    kept = kept[kept["url"].astype(str).str.strip() != ""].reset_index(drop=True)
    kept.to_csv(MURL_EXPECTED_FILE, index=False, encoding="utf-8")
    print(f"[D] Class counts in source: {counts}")
    print(
        f"[D] Wrote {len(kept):,} rows (benign/legitimate/phishing only) -> "
        f"{MURL_EXPECTED_FILE}"
    )
    print(
        "[D] MURL is used for TRAINING AUGMENTATION ONLY; it is never an "
        "evaluation split."
    )
    return True


def download_urlphish(force: bool = False) -> bool:
    """
    Download URL-Phish (Dataset E), the untouched final external evaluation set.

    Returns True if data/raw/URLPhish_urls.csv exists and validates.
    """
    if URLPHISH_EXPECTED_FILE.exists() and not force:
        print(f"[E] URL-Phish already present: {URLPHISH_EXPECTED_FILE}")
        return True

    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    print(f"[E] Downloading {URLPHISH_ARCHIVE_NAME} ...")
    payload = _fetch_verified(URLPHISH_ARCHIVE_URL, URLPHISH_ARCHIVE_SHA256, "E")
    if payload is None:
        print(f"[E] Manual download: {URLPHISH_DATASET_PAGE}")
        return False

    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    if len(df) != URLPHISH_EXPECTED_TOTAL:
        print(f"[E] Expected {URLPHISH_EXPECTED_TOTAL:,} rows, found {len(df):,}.")
        return False
    if "url" not in df.columns or "label" not in df.columns:
        print(f"[E] Expected 'url' and 'label' columns, found {list(df.columns)}")
        return False

    labels = df["label"].dropna().astype(int)
    observed = {int(k): int(v) for k, v in labels.value_counts().items()}
    if any(
        observed.get(raw) != expected
        for raw, expected in URLPHISH_EXPECTED_LABEL_COUNTS.items()
    ):
        print(
            "[E] Label composition does not match the dataset card "
            f"(expected {URLPHISH_EXPECTED_LABEL_COUNTS}, found {observed})."
        )
        return False
    print(
        f"[E] Verified composition: benign(0)={observed[0]:,} "
        f"phishing(1)={observed[1]:,}"
    )

    out = df[["url", "label"]].dropna().reset_index(drop=True)
    out["url"] = out["url"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    out.to_csv(URLPHISH_EXPECTED_FILE, index=False, encoding="utf-8")
    print(f"[E] Wrote {len(out):,} rows -> {URLPHISH_EXPECTED_FILE}")
    print(
        "[E] URL-Phish uses the INTUITIVE convention (0 = benign, 1 = phishing), "
        "the opposite of PHIUSIL/LegitPhish.\n"
        "         It is the FINAL EXTERNAL set and must never be used for training."
    )
    return True


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the raw datasets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing raw files.",
    )
    args = parser.parse_args()

    ok_a = download_mendeley(force=args.force)
    ok_b = download_phiusil()
    ok_c = download_legitphish(force=args.force)
    ok_d = download_murl(force=args.force)
    ok_e = download_urlphish(force=args.force)

    print("\n── Download summary ──────────────────────────────────────")
    print(f"  SMS/Smishing dataset : {'✓ present' if ok_a else '✗ manual download needed'}")
    print(f"  PHIUSIL URL dataset  : {'✓ present' if ok_b else '✗ manual download needed'}")
    print(f"  LegitPhish (dev)     : {'✓ present' if ok_c else '✗ manual download needed'}")
    print(f"  MURL (augmentation)  : {'✓ present' if ok_d else '✗ manual download needed'}")
    print(f"  URL-Phish (external) : {'✓ present' if ok_e else '✗ manual download needed'}")
    print("─────────────────────────────────────────────────────────")
    if not ok_a or not ok_b:
        print("\nComplete the manual download steps above, then re-run inspect_data.py.")
        sys.exit(1)
