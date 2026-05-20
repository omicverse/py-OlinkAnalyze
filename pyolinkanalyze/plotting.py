"""Minimal plotting helpers for Olink NPX analyses.

Two visualisations in v0.1 — both return a ``matplotlib.axes.Axes`` so
they compose with subplot grids:

* :func:`olink_volcano_plot` — log2FC vs -log10 p volcano with assay
  labels.
* :func:`olink_qc_plot` — IQR-based outlier detection on per-sample
  median NPX (subset of the R ``olink_qc_plot``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as _e:  # pragma: no cover
    plt = None
    _MPL_ERR = _e


def _require_mpl():
    if plt is None:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install with "
            "`pip install pyolinkanalyze[plotting]`."
        ) from _MPL_ERR


def olink_volcano_plot(
    stats_df: pd.DataFrame,
    estimate_col: str = "estimate",
    p_col: str = "p.value",
    label_col: str = "Assay",
    threshold: float = 0.05,
    abs_fc_cutoff: float = 0.5,
    n_label: int = 10,
    ax=None,
):
    """Volcano plot from an :func:`olink_ttest` / :func:`olink_wilcox` result.

    Parameters
    ----------
    stats_df :
        DataFrame containing the effect estimate and p-value columns.
    estimate_col :
        Column with the log2 fold-change / NPX difference.
    p_col :
        Column with raw p-value.
    label_col :
        Column to use when annotating top hits.
    threshold :
        BH-adjusted p threshold for the horizontal cutoff (drawn on the
        raw p axis converted to the BH equivalent if available, else
        on raw p).
    abs_fc_cutoff :
        Vertical effect-size cutoff.
    n_label :
        Number of top hits to text-label.
    ax :
        Existing matplotlib axes, or ``None`` to create a new one.
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    df = stats_df.copy()
    df["-log10_p"] = -np.log10(df[p_col].clip(lower=1e-300))
    sig_col = "Adjusted_pval" if "Adjusted_pval" in df.columns else p_col
    df["_sig"] = (df[sig_col] < threshold) & (df[estimate_col].abs() >= abs_fc_cutoff)

    not_sig = df.loc[~df["_sig"]]
    sig = df.loc[df["_sig"]]
    ax.scatter(not_sig[estimate_col], not_sig["-log10_p"],
               s=12, color="lightgrey", alpha=0.6, label="ns")
    ax.scatter(sig[estimate_col], sig["-log10_p"],
               s=18, color="C3", alpha=0.85, label="sig")

    ax.axhline(-np.log10(threshold), color="black", lw=0.6, ls="--")
    ax.axvline(abs_fc_cutoff, color="black", lw=0.4, ls=":")
    ax.axvline(-abs_fc_cutoff, color="black", lw=0.4, ls=":")

    if n_label and label_col in df.columns:
        top = sig.nlargest(n_label, "-log10_p")
        for _, r in top.iterrows():
            ax.text(r[estimate_col], r["-log10_p"], str(r[label_col]),
                    fontsize=7, ha="left", va="bottom")

    ax.set_xlabel(f"effect ({estimate_col})")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.legend(loc="best", frameon=False)
    return ax


def olink_qc_plot(
    df: pd.DataFrame,
    sample_col: str = "SampleID",
    npx_col: str = "NPX",
    iqr_mult: float = 1.5,
    ax=None,
):
    """IQR-based per-sample outlier detection.

    Computes the median NPX per sample, flags samples whose median is
    outside ``[Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR]``, and returns
    a horizontal jitter scatter with outliers highlighted plus a
    ``DataFrame`` of per-sample stats attached as ``ax.qc_stats``.
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    per_sample = df.groupby(sample_col)[npx_col].agg(
        ["median", "mean", "std"]
    ).reset_index()
    q1, q3 = per_sample["median"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    per_sample["outlier"] = (per_sample["median"] < lo) | (per_sample["median"] > hi)

    x = np.arange(len(per_sample))
    ok = ~per_sample["outlier"]
    ax.scatter(x[ok.to_numpy()], per_sample.loc[ok, "median"],
               s=18, color="steelblue", label="pass")
    bad = per_sample["outlier"]
    if bad.any():
        ax.scatter(x[bad.to_numpy()], per_sample.loc[bad, "median"],
                   s=28, color="C3", label="outlier")
        for xi, (_, row) in zip(x[bad.to_numpy()],
                                per_sample.loc[bad].iterrows()):
            ax.text(xi, row["median"], row[sample_col], fontsize=6,
                    ha="left", va="bottom")

    ax.axhline(lo, lw=0.6, color="grey", ls="--")
    ax.axhline(hi, lw=0.6, color="grey", ls="--")
    ax.set_xlabel("sample index")
    ax.set_ylabel(f"per-sample median {npx_col}")
    ax.legend(loc="best", frameon=False)
    ax.qc_stats = per_sample
    return ax
