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

import os
import sys
import pathlib
import textwrap

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Dataset A – Mendeley SMS/Smishing
# ─────────────────────────────────────────────────────────
MENDELEY_EXPECTED_FILE = RAW_DIR / "SMS_dataset.csv"
MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/1/files/"
    "b523ca9a-6e9f-4e34-ac19-e68bca21038d/file_downloaded"
)


def download_mendeley() -> bool:
    """
    Attempt to download the Mendeley SMS dataset programmatically.
    Mendeley does not provide a stable direct-download URL without authentication,
    so this function tries a best-effort GET request and falls back gracefully.

    Returns True if the file is present after the attempt.
    """
    if MENDELEY_EXPECTED_FILE.exists():
        print(f"[A] SMS dataset already present: {MENDELEY_EXPECTED_FILE}")
        return True

    import requests  # noqa: PLC0415

    print("[A] Attempting to download SMS/Smishing dataset from Mendeley ...")
    try:
        resp = requests.get(MENDELEY_URL, timeout=30, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 10_000:
            MENDELEY_EXPECTED_FILE.write_bytes(resp.content)
            print(f"[A] Downloaded → {MENDELEY_EXPECTED_FILE} ({len(resp.content):,} bytes)")
            return True
        else:
            print(f"[A] HTTP {resp.status_code} - automatic download failed "
                  f"(Mendeley requires browser authentication).")
    except Exception as exc:  # noqa: BLE001
        print(f"[A] Request error: {exc}")

    _print_mendeley_manual_instructions()
    return False


def _print_mendeley_manual_instructions() -> None:
    print(
        textwrap.dedent(
            """
        ====================================================================
        MANUAL DOWNLOAD REQUIRED -- SMS/Smishing Dataset (Dataset A)
        ====================================================================
          1. Open a browser and go to:
             https://data.mendeley.com/datasets/f45bkkt8pr/1
          2. Click "Download All" (or download the CSV file).
          3. Rename / place the file at:
             data/raw/SMS_dataset.csv

          Expected columns (verify after download):
            LABEL, TEXT  (or similar -- inspect_data.py will report)
          Expected rows: ~5,971
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
if __name__ == "__main__":
    ok_a = download_mendeley()
    ok_b = download_phiusil()

    print("\n── Download summary ──────────────────────────────────────")
    print(f"  SMS/Smishing dataset : {'✓ present' if ok_a else '✗ manual download needed'}")
    print(f"  PHIUSIL URL dataset  : {'✓ present' if ok_b else '✗ manual download needed'}")
    print("─────────────────────────────────────────────────────────")
    if not ok_a or not ok_b:
        print("\nComplete the manual download steps above, then re-run inspect_data.py.")
        sys.exit(1)
