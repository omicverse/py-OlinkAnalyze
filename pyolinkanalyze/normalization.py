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


# ----------------------------------------------------------------------
# Bridge / subset / N-way normalization sub-modes
# ----------------------------------------------------------------------

def olink_normalization_bridge(
    project_1_df: pd.DataFrame,
    project_2_df: pd.DataFrame,
    bridge_samples,
    project_1_name: str = "P1",
    project_2_name: str = "P2",
    project_ref_name: str = "P1",
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
) -> pd.DataFrame:
    """Bridge normalization — port of ``olink_normalization_bridge``.

    Unlike :func:`olink_normalization` (difference of per-assay
    medians) this matches the R reference exactly: the adjustment
    factor is the **median of the per-bridge-sample NPX differences**
    (a paired statistic). ``project_2_df`` is shifted onto
    ``project_1_df`` when ``project_ref_name == project_1_name``.

    Parameters
    ----------
    bridge_samples :
        Either an iterable of shared SampleIDs (same names in both
        projects) or a dict ``{'DF1': [...], 'DF2': [...]}`` mapping
        the bridge IDs of project 1 to those of project 2.
    """
    if isinstance(bridge_samples, dict):
        ids1 = list(bridge_samples["DF1"])
        ids2 = list(bridge_samples["DF2"])
    else:
        ids1 = ids2 = list(bridge_samples)
    if len(ids1) != len(ids2):
        raise ValueError("DF1 and DF2 bridge lists must be equal length.")

    ref_is_p1 = project_ref_name == project_1_name
    if not ref_is_p1 and project_ref_name != project_2_name:
        raise ValueError(
            "project_ref_name must equal project_1_name or project_2_name."
        )

    id_map = dict(zip(ids2, ids1))
    p1 = project_1_df.loc[project_1_df[sample_col].isin(ids1),
                          [sample_col, assay_col, npx_col]].copy()
    p2 = project_2_df.loc[project_2_df[sample_col].isin(ids2),
                          [sample_col, assay_col, npx_col]].copy()
    p2["_paired"] = p2[sample_col].map(id_map)
    p1 = p1.rename(columns={sample_col: "_paired"})

    merged = p1.merge(p2, on=["_paired", assay_col],
                      suffixes=("_p1", "_p2"))
    if ref_is_p1:
        merged["Diff"] = merged[f"{npx_col}_p1"] - merged[f"{npx_col}_p2"]
    else:
        merged["Diff"] = merged[f"{npx_col}_p2"] - merged[f"{npx_col}_p1"]
    adj = merged.groupby(assay_col)["Diff"].median()

    target_df = project_2_df if ref_is_p1 else project_1_df
    ref_df = project_1_df if ref_is_p1 else project_2_df
    target_name = project_2_name if ref_is_p1 else project_1_name

    adj = adj.reindex(target_df[assay_col].unique()).fillna(0.0)
    tgt = target_df.copy()
    tgt["Adj_factor"] = tgt[assay_col].map(adj).astype(float)
    tgt[npx_col] = tgt[npx_col].astype(float) + tgt["Adj_factor"]
    tgt["Project"] = target_name
    if "LOD" in tgt.columns:
        tgt["LOD"] = tgt["LOD"].astype(float) + tgt["Adj_factor"]

    ref = ref_df.copy()
    ref["Adj_factor"] = 0.0
    ref["Project"] = project_ref_name
    return pd.concat([ref, tgt], axis=0, ignore_index=True)


def olink_normalization_subset(
    project_1_df: pd.DataFrame,
    project_2_df: pd.DataFrame,
    reference_samples,
    project_1_name: str = "P1",
    project_2_name: str = "P2",
    project_ref_name: str = "P1",
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
) -> pd.DataFrame:
    """Subset normalization — port of ``olink_normalization_subset``.

    Used when the two projects share no bridging samples but each
    contains a comparable representative subset (e.g. a shared sample
    type). The adjustment factor is the **difference of per-assay
    medians** computed over the (independent) reference subsets — this
    is the ``MOD_FLAG=FALSE`` branch of the R ``olink_normalization``.

    ``reference_samples`` is an iterable of shared IDs or a dict
    ``{'DF1': [...], 'DF2': [...]}``.
    """
    if isinstance(reference_samples, dict):
        ids1 = list(reference_samples["DF1"])
        ids2 = list(reference_samples["DF2"])
    else:
        ids1 = ids2 = list(reference_samples)

    ref_is_p1 = project_ref_name == project_1_name
    if not ref_is_p1 and project_ref_name != project_2_name:
        raise ValueError(
            "project_ref_name must equal project_1_name or project_2_name."
        )

    sub1 = project_1_df.loc[project_1_df[sample_col].isin(ids1)]
    sub2 = project_2_df.loc[project_2_df[sample_col].isin(ids2)]
    med1 = sub1.groupby(assay_col)[npx_col].median()
    med2 = sub2.groupby(assay_col)[npx_col].median()

    target_df = project_2_df if ref_is_p1 else project_1_df
    ref_df = project_1_df if ref_is_p1 else project_2_df
    target_name = project_2_name if ref_is_p1 else project_1_name
    med_ref = med1 if ref_is_p1 else med2
    med_tgt = med2 if ref_is_p1 else med1

    adj = (med_ref - med_tgt).rename("Adj_factor")
    adj = adj.reindex(target_df[assay_col].unique()).fillna(0.0)

    tgt = target_df.copy()
    tgt["Adj_factor"] = tgt[assay_col].map(adj).astype(float)
    tgt[npx_col] = tgt[npx_col].astype(float) + tgt["Adj_factor"]
    tgt["Project"] = target_name
    if "LOD" in tgt.columns:
        tgt["LOD"] = tgt["LOD"].astype(float) + tgt["Adj_factor"]

    ref = ref_df.copy()
    ref["Adj_factor"] = 0.0
    ref["Project"] = project_ref_name
    return pd.concat([ref, tgt], axis=0, ignore_index=True)


def olink_normalization_n(norm_schema, npx_col: str = "NPX",
                          assay_col: str = "OlinkID",
                          sample_col: str = "SampleID") -> pd.DataFrame:
    """N-way (multi-batch) normalization — port of ``olink_normalization_n``.

    Normalizes a chain / tree of >= 3 projects onto a global reference.

    ``norm_schema`` is a list of dicts, one per project, each with:

    * ``name`` — project label.
    * ``data`` — the project's NPX DataFrame.
    * ``order`` — integer; ``order == 1`` is the global reference.
    * ``normalization_type`` — ``'Bridge'`` or ``'Subset'`` (ignored
      for the reference).
    * ``normalize_to`` — name of the already-normalized project to
      anchor to (ignored for the reference).
    * ``samples`` — bridge / reference sample list or
      ``{'DF1': ..., 'DF2': ...}`` dict.

    Returns the concatenation of all projects on the common scale, with
    ``Project`` and ``Adj_factor`` columns.
    """
    schema = sorted(norm_schema, key=lambda d: d["order"])
    ref = schema[0]
    normalized: dict = {}
    ref_df = ref["data"].copy()
    ref_df["Project"] = ref["name"]
    ref_df["Adj_factor"] = 0.0
    normalized[ref["name"]] = ref_df

    for entry in schema[1:]:
        anchor_name = entry["normalize_to"]
        anchor = normalized[anchor_name].drop(
            columns=[c for c in ("Project", "Adj_factor")
                     if c in normalized[anchor_name].columns]
        )
        ntype = entry.get("normalization_type", "Bridge")
        fn = (olink_normalization_bridge if ntype == "Bridge"
              else olink_normalization_subset)
        kw = dict(project_1_name=anchor_name, project_2_name=entry["name"],
                  project_ref_name=anchor_name, npx_col=npx_col,
                  sample_col=sample_col, assay_col=assay_col)
        if ntype == "Bridge":
            res = fn(anchor, entry["data"], entry["samples"], **kw)
        else:
            res = fn(anchor, entry["data"], entry["samples"], **kw)
        normalized[entry["name"]] = res.loc[res["Project"] == entry["name"]]

    return pd.concat(list(normalized.values()), axis=0, ignore_index=True)


def olink_bridge_selector(
    df: pd.DataFrame,
    sample_missing_freq: float = 0.1,
    n: int = 8,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
    panel_col: str = "Panel",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Pick optimal bridging samples — port of ``olink_bridgeselector``.

    Mirrors the R algorithm:

    1. Drop control samples and QC-warning samples; flag IQR / median
       outliers per panel (mean ± 3 SD) and drop them.
    2. Set NPX <= LOD to missing; compute per-sample missing frequency.
    3. Keep samples with missing frequency below ``sample_missing_freq``.
    4. Of those, pick ``n`` samples evenly spaced along the
       mean-NPX-sorted ranking so the bridges span the dynamic range.

    Returns a DataFrame ``SampleID, PercAssaysBelowLOD, MeanNPX``.
    """
    rng = np.random.default_rng(seed)
    work = df.copy()

    # Drop control samples.
    work = work.loc[~work[sample_col].astype(str)
                    .str.upper().str.contains("CONTROL_SAMPLE", na=False)]

    has_panel = panel_col in work.columns
    grp_keys = [panel_col, sample_col] if has_panel else [sample_col]

    # Per-(panel,)sample IQR + median.
    per = (work.groupby(grp_keys)[npx_col]
           .agg(IQR=lambda s: s.quantile(0.75) - s.quantile(0.25),
                sample_median="median")
           .reset_index())
    pgrp = per.groupby(panel_col) if has_panel else [(None, per)]
    flagged = []
    for _, g in pgrp:
        g = g.copy()
        m_med, s_med = g["sample_median"].mean(), g["sample_median"].std()
        m_iqr, s_iqr = g["IQR"].mean(), g["IQR"].std()
        g["Outlier"] = (~(
            (g["sample_median"] < m_med + 3 * s_med)
            & (g["sample_median"] > m_med - 3 * s_med)
            & (g["IQR"] > m_iqr - 3 * s_iqr)
            & (g["IQR"] < m_iqr + 3 * s_iqr)
        )).astype(int)
        flagged.append(g)
    per = pd.concat(flagged, axis=0)
    outlier_samples = set(
        per.loc[per["Outlier"] == 1, sample_col].astype(str)
    )

    # QC warnings.
    work["_qc"] = (work["QC_Warning"].astype(str).str.upper()
                   if "QC_Warning" in work.columns else "PASS")
    qc_pass = work.groupby(sample_col)["_qc"].apply(
        lambda s: (s == "PASS").all()
    )
    pass_samples = set(qc_pass.index[qc_pass].astype(str))

    # NPX <= LOD -> missing.
    npx = pd.to_numeric(work[npx_col], errors="coerce")
    if "LOD" in work.columns:
        below = npx <= pd.to_numeric(work["LOD"], errors="coerce")
    else:
        below = pd.Series(False, index=work.index)
    npx_masked = npx.where(~below, np.nan)
    work = work.assign(_npx_masked=npx_masked)

    keep = []
    for sid, g in work.groupby(sample_col):
        sid = str(sid)
        if sid not in pass_samples or sid in outlier_samples:
            continue
        perc = float(g["_npx_masked"].isna().mean())
        if perc < sample_missing_freq:
            keep.append({
                sample_col: sid,
                "PercAssaysBelowLOD": perc,
                "MeanNPX": float(g["_npx_masked"].mean()),
            })
    cand = pd.DataFrame(keep)
    if len(cand) < n:
        raise ValueError(
            f"Only {len(cand)} samples qualify; increase "
            f"sample_missing_freq or decrease n."
        )
    if len(cand) == n:
        return cand.reset_index(drop=True)

    cand = cand.sort_values("MeanNPX", ascending=False).reset_index(drop=True)
    # Evenly spaced indices spanning the range (R: floor(seq) interior).
    idx = np.floor(np.linspace(0, len(cand) - 1, n + 2)[1:-1]).astype(int)
    selected = cand.iloc[idx].copy()
    selected = selected.iloc[rng.permutation(len(selected))]
    return selected.reset_index(drop=True)
