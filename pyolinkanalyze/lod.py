"""Limit-of-Detection (LOD) handling for Olink NPX data.

Port of ``OlinkAnalyze::olink_lod``.

OlinkAnalyze 5.x derives the LOD from the raw Olink count data
(negative-control / fixed-LOD reference tables shipped with the
panel). Those binary inputs are not available in an NPX-only workflow,
so :func:`olink_lod` here operates on the NPX long-format directly:

* ``lod_method='NCLOD'`` — *negative-control* LOD. For each assay the
  LOD is estimated from the NPX of negative-control / blank samples
  (``LOD = median(NC) + 3 * MAD(NC)`` — the standard 3-sigma rule). If
  no NC samples are present we fall back to the per-assay NPX
  ``min - epsilon`` so nothing is flagged spuriously.
* ``lod_method='FixedLOD'`` — read a fixed per-assay LOD table from
  ``lod_file`` (a 2-column ``OlinkID;LOD`` semicolon CSV, matching
  Olink's fixed-LOD file format).

In both cases the function adds (or overwrites) a ``LOD`` column and a
``below_LOD`` flag (``True`` where ``NPX <= LOD``), and reports the
per-assay missing-frequency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


_NC_TOKENS = ("NEGATIVE_CONTROL", "NEG_CONTROL", "NEGATIVECONTROL",
              "NC", "BLANK", "CONTROL_SAMPLE")


def _is_nc(sample_id: str) -> bool:
    s = str(sample_id).upper().replace(" ", "_")
    return any(tok in s for tok in _NC_TOKENS)


def olink_lod(
    df: pd.DataFrame,
    lod_file: Optional[Union[str, Path]] = None,
    lod_method: str = "NCLOD",
    sample_type_col: Optional[str] = None,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
    sd_mult: float = 3.0,
) -> pd.DataFrame:
    """Compute a per-assay LOD and flag NPX values below it.

    Parameters
    ----------
    df :
        Long-format NPX frame.
    lod_file :
        Path to a semicolon-separated ``OlinkID;LOD`` table — required
        when ``lod_method='FixedLOD'``.
    lod_method :
        ``'NCLOD'`` (default, negative-control estimate) or
        ``'FixedLOD'`` (external table).
    sample_type_col :
        Optional column whose value identifies negative-control samples
        (any value containing ``NC`` / ``CONTROL`` / ``BLANK``). If
        ``None``, NC detection falls back to the ``SampleID`` string.
    sd_mult :
        Multiplier of the robust SD in the NC-LOD rule (Olink uses 3).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``LOD`` and ``below_LOD`` (bool) columns.
        ``PercAssaysBelowLOD`` per assay is attached as
        ``result.attrs['lod_summary']``.
    """
    out = df.copy()

    if lod_method == "FixedLOD":
        if lod_file is None:
            raise ValueError(
                "lod_file must be specified for lod_method='FixedLOD'."
            )
        lod_tbl = pd.read_csv(lod_file, sep=";")
        # Be permissive about column names.
        cols = {c.lower(): c for c in lod_tbl.columns}
        oid_c = cols.get("olinkid", lod_tbl.columns[0])
        lod_c = cols.get("lod", lod_tbl.columns[-1])
        lod_map = dict(zip(lod_tbl[oid_c].astype(str),
                           pd.to_numeric(lod_tbl[lod_c], errors="coerce")))
        out["LOD"] = out[assay_col].astype(str).map(lod_map)
    elif lod_method == "NCLOD":
        if sample_type_col is not None and sample_type_col in out.columns:
            nc_mask = out[sample_type_col].astype(str).str.upper().apply(
                lambda s: any(t in s for t in _NC_TOKENS)
            )
        else:
            nc_mask = out[sample_col].apply(_is_nc)

        nc = out.loc[nc_mask]
        lod_map = {}
        for oid, grp in out.groupby(assay_col, sort=False):
            nc_vals = nc.loc[nc[assay_col] == oid, npx_col].to_numpy(dtype=float)
            nc_vals = nc_vals[~np.isnan(nc_vals)]
            if len(nc_vals) >= 1:
                med = float(np.median(nc_vals))
                # Robust SD via MAD; fall back to plain SD for n>=2.
                mad = float(np.median(np.abs(nc_vals - med)))
                rsd = 1.4826 * mad
                if rsd == 0.0 and len(nc_vals) >= 2:
                    rsd = float(np.std(nc_vals, ddof=1))
                lod_map[oid] = med + sd_mult * rsd
            else:
                vals = grp[npx_col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                lod_map[oid] = (float(np.min(vals)) - 1e-9
                                if len(vals) else np.nan)
        out["LOD"] = out[assay_col].map(lod_map).astype(float)
    else:
        raise ValueError("lod_method must be 'NCLOD' or 'FixedLOD'.")

    npx = pd.to_numeric(out[npx_col], errors="coerce")
    out["below_LOD"] = (npx <= out["LOD"]).fillna(False)

    summary = (
        out.groupby(assay_col)
        .agg(LOD=("LOD", "first"),
             n=(npx_col, "size"),
             n_below=("below_LOD", "sum"))
        .reset_index()
    )
    summary["PercAssaysBelowLOD"] = summary["n_below"] / summary["n"]
    out.attrs["lod_summary"] = summary
    return out
