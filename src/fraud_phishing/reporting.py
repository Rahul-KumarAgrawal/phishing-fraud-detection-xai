"""
reporting.py
============
Helpers for the generated report artefacts.

``reports/model_comparison.csv`` is written by *both* training scripts. It used to
be appended to, so re-running a trainer duplicated its rows and rows from
different runs of the same branch became indistinguishable.
``update_model_comparison`` replaces the rows belonging to one branch while
preserving the other branch's rows, so the file always reflects the most recent
run of each branch exactly once.
"""

import csv
import pathlib

FIELDNAMES = ["branch", "model", "val_f1", "val_roc_auc", "threshold", "selected"]


def update_model_comparison(
    path: pathlib.Path | str,
    branch: str,
    rows: list[dict],
) -> None:
    """
    Replace the rows for ``branch`` in ``path``, keeping other branches intact.

    Parameters
    ----------
    path:
        Location of ``model_comparison.csv``.
    branch:
        Branch key to replace, e.g. ``"url"`` or ``"message"``.
    rows:
        Rows for this branch; keys outside :data:`FIELDNAMES` are ignored.
    """
    path = pathlib.Path(path)

    kept: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            kept = [
                row for row in csv.DictReader(handle) if row.get("branch") != branch
            ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in kept:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
