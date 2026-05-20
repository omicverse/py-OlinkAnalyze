"""Bridge / reference normalization for Olink NPX data.

Olink projects routinely span multiple plates / batches. The standard
recommended batch-correction approach is **bridge normalization** — a
small set of overlapping samples is run on both projects, and the
median NPX difference per assay (over the bridging samples) is
subtracted from the target project so the two anchor to a common scale.

R reference: ``OlinkAnalyze::olink_normalization`` with
``reference_medians = NULL`` (i.e. bridge mode).

Algorithm (per OlinkID):

1. For each assay, compute the median NPX in the bridging samples on
   df_ref and df_target.
2. ``adj = median_ref - median_target``.
3. Apply ``NPX_norm = NPX + adj`` to *all* rows of df_target whose
   OlinkID matches.

Assays present only in df_ref or only in df_target are passed through
unchanged with ``Adj_factor = 0``. The returned DataFrame has the same
schema as the inputs, plus a ``Project`` and ``Adj_factor`` column.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def olink_normalization(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    overlapping_samples_df1: Optional[Iterable[str]] = None,
    overlapping_samples_df2: Optional[Iterable[str]] = None,
    reference_project: str = "ref",
    target_project: str = "target",
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
) -> pd.DataFrame:
    """Bridge-normalize ``df_target`` onto ``df_ref``.

    Parameters
    ----------
    df_ref, df_target :
        Long-format Olink NPX frames as returned by
        :func:`pyolinkanalyze.io.read_npx_csv`.
    overlapping_samples_df1 :
        Sample IDs in ``df_ref`` that are the bridging samples.
    overlapping_samples_df2 :
        Sample IDs in ``df_target`` corresponding 1-to-1 to
        ``overlapping_samples_df1``. If ``None``, assumed equal to
        ``overlapping_samples_df1`` (same sample names in both
        projects).
    reference_project, target_project :
        Strings to write into the output's ``Project`` column.
    npx_col, sample_col, assay_col :
        Column names — override if your NPX export uses non-standard
        labels.

    Returns
    -------
    pandas.DataFrame
        Concatenation of ``df_ref`` (unchanged) and ``df_target``
        (with ``NPX`` column shifted by the per-assay
        ``Adj_factor``). Adds columns ``Project`` and
        ``Adj_factor``.
    """
    if overlapping_samples_df1 is None:
        raise ValueError(
            "overlapping_samples_df1 must be provided (bridging samples)."
        )
    bridge_ref = list(overlapping_samples_df1)
    bridge_tgt = list(overlapping_samples_df2 if overlapping_samples_df2 is not None
                      else bridge_ref)
    if len(bridge_ref) != len(bridge_tgt):
        raise ValueError(
            "overlapping_samples_df1 and overlapping_samples_df2 must have "
            f"the same length (got {len(bridge_ref)} and {len(bridge_tgt)})."
        )

    # Per-assay median NPX on bridging samples in each project.
    ref_bridge = df_ref.loc[df_ref[sample_col].isin(bridge_ref)]
    tgt_bridge = df_target.loc[df_target[sample_col].isin(bridge_tgt)]
    med_ref = ref_bridge.groupby(assay_col)[npx_col].median()
    med_tgt = tgt_bridge.groupby(assay_col)[npx_col].median()

    # Adjustment factor per assay; NaN where one side has no bridging
    # data — those assays keep adj=0.
    adj = (med_ref - med_tgt).rename("Adj_factor")
    adj = adj.reindex(df_target[assay_col].unique()).fillna(0.0)

    df_target_norm = df_target.copy()
    df_target_norm["Adj_factor"] = df_target_norm[assay_col].map(adj).astype(float)
    df_target_norm[npx_col] = (
        df_target_norm[npx_col].astype(float)
        + df_target_norm["Adj_factor"]
    )
    df_target_norm["Project"] = target_project

    df_ref_out = df_ref.copy()
    df_ref_out["Adj_factor"] = 0.0
    df_ref_out["Project"] = reference_project

    out = pd.concat([df_ref_out, df_target_norm], axis=0, ignore_index=True)
    return out


def olink_normalization_reference_medians(
    df_target: pd.DataFrame,
    reference_medians: pd.Series,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    target_project: str = "target",
) -> pd.DataFrame:
    """Normalize ``df_target`` to externally supplied per-assay medians.

    Reference-median mode of ``OlinkAnalyze::olink_normalization`` —
    use when you don't have bridging samples but you do have a public
    reference distribution (e.g. Olink's calibration medians).
    """
    if not isinstance(reference_medians, pd.Series):
        reference_medians = pd.Series(reference_medians)

    med_tgt = df_target.groupby(assay_col)[npx_col].median()
    common = reference_medians.index.intersection(med_tgt.index)
    adj = (reference_medians.loc[common] - med_tgt.loc[common]).rename("Adj_factor")
    adj = adj.reindex(df_target[assay_col].unique()).fillna(0.0)

    out = df_target.copy()
    out["Adj_factor"] = out[assay_col].map(adj).astype(float)
    out[npx_col] = out[npx_col].astype(float) + out["Adj_factor"]
    out["Project"] = target_project
    return out
