"""NPX long-format CSV readers.

Olink's Explore / Target panels ship NPX (Normalized Protein eXpression)
data as a *long* table — one row per (SampleID, OlinkID). The canonical
columns we rely on are:

* ``SampleID``     — sample identifier.
* ``OlinkID``      — assay/probe identifier (e.g. ``OID00123``).
* ``UniProt``      — UniProt accession of the target.
* ``Assay``        — gene name / assay label.
* ``Panel``        — panel name (e.g. ``Inflammation``).
* ``NPX``          — log2-scale normalised expression.
* ``QC_Warning``   — per-sample QC flag (``PASS`` / ``WARN``).
* ``LOD``          — limit of detection (NPX scale).

Older Target-96 exports use semicolon-separated CSV with a one-row
metadata header; newer Explore CSV exports are comma-separated. We
auto-detect both via ``pandas.read_csv``'s sniffer.

The reader is permissive: missing optional columns are tolerated and
``NPX`` / ``LOD`` are coerced to ``float``.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Union

import pandas as pd


_REQUIRED = ("SampleID", "OlinkID", "NPX")
_OPTIONAL = ("UniProt", "Assay", "Panel", "QC_Warning", "LOD",
             "MissingFreq", "Normalization")


def read_npx_csv(
    path: Union[str, Path],
    sep: Union[str, None] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read an Olink long-format NPX CSV into a tidy DataFrame.

    Parameters
    ----------
    path :
        Path to the NPX CSV (semicolon- or comma-separated).
    sep :
        Field separator. ``None`` (default) lets pandas auto-detect.
    encoding :
        File encoding. Default ``utf-8`` — set to ``latin-1`` for
        legacy NPX Manager exports.

    Returns
    -------
    pandas.DataFrame
        Tidy long-form table with at least
        ``SampleID``, ``OlinkID``, ``NPX``. ``NPX`` and ``LOD`` are
        coerced to ``float``; ``QC_Warning`` is upper-cased.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if sep is None:
        # Sniff the separator from the first non-empty line.
        with open(path, "r", encoding=encoding) as f:
            head = f.readline()
        sep = ";" if head.count(";") > head.count(",") else ","

    df = pd.read_csv(path, sep=sep, encoding=encoding,
                     low_memory=False)
    return _normalize_npx_frame(df)


def read_npx_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce / validate a long-format NPX DataFrame in memory.

    Useful when NPX is generated programmatically (e.g. in tests).
    """
    return _normalize_npx_frame(df.copy())


def _normalize_npx_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"NPX frame missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    df["SampleID"] = df["SampleID"].astype(str)
    df["OlinkID"] = df["OlinkID"].astype(str)
    df["NPX"] = pd.to_numeric(df["NPX"], errors="coerce")

    if "LOD" in df.columns:
        df["LOD"] = pd.to_numeric(df["LOD"], errors="coerce")
    if "QC_Warning" in df.columns:
        df["QC_Warning"] = df["QC_Warning"].astype(str).str.upper()
    return df


def write_npx_csv(df: pd.DataFrame, path: Union[str, Path],
                  sep: str = ",") -> None:
    """Convenience writer — round-trip with :func:`read_npx_csv`."""
    df.to_csv(path, sep=sep, index=False)
