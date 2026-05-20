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


# ----------------------------------------------------------------------
# Olink color palette + theme — port of olink_pal / set_plot_theme etc.
# ----------------------------------------------------------------------

# The 10 Olink "crispy" brand colors, in the order olink_pal() returns
# them for small n (turquoise, red, darkblue, yellow, teal, pink,
# green, purple, orange, lightblue).
OLINK_COLORS_ORDERED = [
    "#00C7E1", "#FE1F04", "#00559E", "#FFC700", "#077183",
    "#FF51B8", "#27AE55", "#6A27AE", "#FF8C22", "#A2D9F5",
]
# Ramp order used by olink_pal for large n.
OLINK_COLORS_RAMP = [
    "#FE1F04", "#FF8C22", "#FFC700", "#27AE55", "#077183",
    "#00C7E1", "#A2D9F5", "#00559E", "#6A27AE", "#FF51B8",
]


def olink_pal(n: int = 10):
    """Return ``n`` colors from the Olink brand palette.

    Port of ``OlinkAnalyze::olink_pal``. For ``n <= 10`` the fixed
    ordered palette is used; for larger ``n`` colors are interpolated
    along the ramp.
    """
    if n <= len(OLINK_COLORS_ORDERED):
        return OLINK_COLORS_ORDERED[:n]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("olink", OLINK_COLORS_RAMP)
    return [_to_hex(cmap(i / (n - 1))) for i in range(n)]


def _to_hex(rgba):
    from matplotlib.colors import to_hex
    return to_hex(rgba)


def olink_color_discrete(n: int = 10):
    """List of discrete Olink colors (alias of :func:`olink_pal`)."""
    return olink_pal(n)


def olink_fill_discrete(n: int = 10):
    """List of discrete Olink fill colors (alias of :func:`olink_pal`)."""
    return olink_pal(n)


def olink_color_gradient():
    """Continuous Olink colormap (red -> blue) for color scales."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "olink_gradient", ["#FE1F04", "#FFC700", "#00C7E1", "#00559E"])


def olink_fill_gradient():
    """Continuous Olink colormap for fill scales."""
    return olink_color_gradient()


def set_plot_theme(ax=None, base_size: int = 11):
    """Apply the Olink plot theme to ``ax`` (or current axes).

    Light port of ``OlinkAnalyze::set_plot_theme`` — a minimal,
    de-cluttered theme with a light grid and no top/right spines.
    """
    _require_mpl()
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=base_size - 1)
    return ax


# ----------------------------------------------------------------------
# Distribution / boxplot
# ----------------------------------------------------------------------

def olink_boxplot(
    df: pd.DataFrame,
    variable: str,
    olinkids,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    label_col: str = "Assay",
    ax=None,
):
    """Boxplot of NPX by ``variable`` for the chosen assay(s).

    Port of ``OlinkAnalyze::olink_boxplot``. Pass one or several
    OlinkIDs; each is drawn as a group of boxes (one per level of
    ``variable``).
    """
    _require_mpl()
    if isinstance(olinkids, str):
        olinkids = [olinkids]
    if ax is None:
        _, ax = plt.subplots(figsize=(2 + 1.5 * len(olinkids), 4))

    levels = sorted(df[variable].dropna().unique())
    colors = olink_pal(len(levels))
    width = 0.8 / max(len(levels), 1)
    positions, ticklabels = [], []
    for gi, oid in enumerate(olinkids):
        sub = df.loc[df[assay_col] == oid]
        for li, lv in enumerate(levels):
            vals = sub.loc[sub[variable] == lv, npx_col].dropna().to_numpy()
            pos = gi + (li - (len(levels) - 1) / 2) * width
            if len(vals):
                bp = ax.boxplot([vals], positions=[pos], widths=width * 0.9,
                                patch_artist=True, showfliers=False)
                for box in bp["boxes"]:
                    box.set_facecolor(colors[li])
                    box.set_alpha(0.8)
        positions.append(gi)
        label = oid
        if label_col in sub.columns and len(sub):
            label = str(sub[label_col].iloc[0])
        ticklabels.append(label)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right")
    ax.set_ylabel(npx_col)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.8)
               for i in range(len(levels))]
    ax.legend(handles, [str(x) for x in levels], title=variable,
              frameon=False, fontsize=8)
    set_plot_theme(ax)
    return ax


def olink_dist_plot(
    df: pd.DataFrame,
    color_by: Optional[str] = None,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    ax=None,
):
    """Per-sample NPX distribution boxplot.

    Port of ``OlinkAnalyze::olink_dist_plot`` — one box per sample,
    optionally colored by a grouping column.
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    samples = list(df[sample_col].drop_duplicates())
    color_map = {}
    if color_by is not None and color_by in df.columns:
        groups = sorted(df[color_by].dropna().unique())
        pal = olink_pal(len(groups))
        gcol = dict(zip(groups, pal))
        for s in samples:
            g = df.loc[df[sample_col] == s, color_by].iloc[0]
            color_map[s] = gcol.get(g, "steelblue")
    data = [df.loc[df[sample_col] == s, npx_col].dropna().to_numpy()
            for s in samples]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for s, box in zip(samples, bp["boxes"]):
        box.set_facecolor(color_map.get(s, "steelblue"))
        box.set_alpha(0.8)
    ax.set_xticks([])
    ax.set_xlabel("sample")
    ax.set_ylabel(npx_col)
    set_plot_theme(ax)
    return ax


# ----------------------------------------------------------------------
# PCA / UMAP / heatmap
# ----------------------------------------------------------------------

def _npx_wide(df, sample_col, assay_col, npx_col):
    """Long NPX -> samples x assays wide matrix (mean-imputed)."""
    wide = df.pivot_table(index=sample_col, columns=assay_col,
                          values=npx_col, aggfunc="mean")
    wide = wide.fillna(wide.mean())
    return wide


def olink_pca_plot(
    df: pd.DataFrame,
    color_by: Optional[str] = None,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
    n_components: int = 2,
    ax=None,
):
    """PCA scatter of samples in NPX space.

    Port of ``OlinkAnalyze::olink_pca_plot`` — standardizes assays,
    runs :class:`sklearn.decomposition.PCA` and plots PC1 vs PC2. The
    PCA result is attached as ``ax.pca_scores``.
    """
    _require_mpl()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    wide = _npx_wide(df, sample_col, assay_col, npx_col)
    X = StandardScaler().fit_transform(wide.to_numpy())
    pca = PCA(n_components=max(2, n_components))
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(
        scores[:, :2], columns=["PC1", "PC2"], index=wide.index)

    if color_by is not None and color_by in df.columns:
        meta = df.drop_duplicates(sample_col).set_index(sample_col)[color_by]
        groups = sorted(meta.dropna().unique())
        pal = olink_pal(len(groups))
        for g, col in zip(groups, pal):
            sids = meta.index[meta == g]
            sel = scores_df.loc[scores_df.index.isin(sids)]
            ax.scatter(sel["PC1"], sel["PC2"], s=30, color=col,
                       label=str(g), alpha=0.85)
        ax.legend(title=color_by, frameon=False, fontsize=8)
    else:
        ax.scatter(scores_df["PC1"], scores_df["PC2"], s=30,
                   color=OLINK_COLORS_ORDERED[0], alpha=0.85)
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.pca_scores = scores_df
    set_plot_theme(ax)
    return ax


def olink_umap_plot(
    df: pd.DataFrame,
    color_by: Optional[str] = None,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
    ax=None,
    random_state: int = 0,
):
    """UMAP embedding of samples in NPX space.

    Port of ``OlinkAnalyze::olink_umap_plot``. Uses ``umap-learn`` if
    installed, otherwise falls back to a 2-component PCA with a warning.
    The embedding is attached as ``ax.umap_scores``.
    """
    _require_mpl()
    import warnings as _w
    from sklearn.preprocessing import StandardScaler

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    wide = _npx_wide(df, sample_col, assay_col, npx_col)
    X = StandardScaler().fit_transform(wide.to_numpy())
    try:
        import umap  # type: ignore
        emb = umap.UMAP(random_state=random_state,
                        n_neighbors=min(15, len(wide) - 1)).fit_transform(X)
        xlab, ylab = "UMAP1", "UMAP2"
    except Exception:
        _w.warn("umap-learn not available; falling back to PCA.")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(X)
        xlab, ylab = "PC1 (UMAP fallback)", "PC2 (UMAP fallback)"
    emb_df = pd.DataFrame(emb, columns=["c1", "c2"], index=wide.index)

    if color_by is not None and color_by in df.columns:
        meta = df.drop_duplicates(sample_col).set_index(sample_col)[color_by]
        groups = sorted(meta.dropna().unique())
        pal = olink_pal(len(groups))
        for g, col in zip(groups, pal):
            sids = meta.index[meta == g]
            sel = emb_df.loc[emb_df.index.isin(sids)]
            ax.scatter(sel["c1"], sel["c2"], s=30, color=col,
                       label=str(g), alpha=0.85)
        ax.legend(title=color_by, frameon=False, fontsize=8)
    else:
        ax.scatter(emb_df["c1"], emb_df["c2"], s=30,
                   color=OLINK_COLORS_ORDERED[0], alpha=0.85)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.umap_scores = emb_df
    set_plot_theme(ax)
    return ax


def olink_heatmap_plot(
    df: pd.DataFrame,
    npx_col: str = "NPX",
    sample_col: str = "SampleID",
    assay_col: str = "OlinkID",
    z_score: bool = True,
    cmap=None,
    ax=None,
):
    """NPX heatmap (assays x samples).

    Port of ``OlinkAnalyze::olink_heatmap_plot``. By default each assay
    is z-scored across samples. Returns the axes; the matrix is
    attached as ``ax.heatmap_data``.
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    wide = _npx_wide(df, sample_col, assay_col, npx_col)
    mat = wide.T  # assays x samples
    if z_score:
        mat = mat.sub(mat.mean(axis=1), axis=0).div(
            mat.std(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(mat.to_numpy(), aspect="auto",
                   cmap=cmap or olink_color_gradient())
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=6)
    ax.set_xticks([])
    ax.set_xlabel("samples")
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                       label="z-score" if z_score else npx_col)
    ax.heatmap_data = mat
    return ax


def olink_lmer_plot(
    df: pd.DataFrame,
    variable: str,
    random: str,
    olinkids,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    label_col: str = "Assay",
    ax=None,
):
    """Per-subject NPX trajectory plot for LMM assays.

    Port of ``OlinkAnalyze::olink_lmer_plot``. For each chosen assay,
    draws every subject's NPX across the levels of ``variable`` (faint
    lines) plus the group means (bold), the standard mixed-model
    diagnostic visual.
    """
    _require_mpl()
    if isinstance(olinkids, str):
        olinkids = [olinkids]
    if ax is None:
        _, ax = plt.subplots(1, len(olinkids),
                             figsize=(3.5 * len(olinkids), 4),
                             squeeze=False)
        axes = ax[0]
    else:
        axes = [ax] if not hasattr(ax, "__len__") else list(ax)

    levels = sorted(df[variable].dropna().unique())
    xpos = {lv: i for i, lv in enumerate(levels)}
    for axi, oid in zip(axes, olinkids):
        sub = df.loc[df[assay_col] == oid]
        for subj, g in sub.groupby(random):
            piv = g.groupby(variable)[npx_col].mean().reindex(levels)
            axi.plot([xpos[lv] for lv in levels], piv.to_numpy(),
                     color="0.7", lw=0.8, alpha=0.6, marker="o", ms=3)
        means = sub.groupby(variable)[npx_col].mean().reindex(levels)
        axi.plot([xpos[lv] for lv in levels], means.to_numpy(),
                 color=OLINK_COLORS_ORDERED[1], lw=2.2, marker="o", ms=6)
        axi.set_xticks(range(len(levels)))
        axi.set_xticklabels([str(x) for x in levels])
        axi.set_xlabel(variable)
        axi.set_ylabel(npx_col)
        label = oid
        if label_col in sub.columns and len(sub):
            label = str(sub[label_col].iloc[0])
        axi.set_title(label, fontsize=9)
        set_plot_theme(axi)
    return axes[0] if len(axes) == 1 else axes


# ----------------------------------------------------------------------
# Pathway plots — port of olink_pathway_heatmap / olink_pathway_visualization
# ----------------------------------------------------------------------

def olink_pathway_heatmap(
    enrichment: pd.DataFrame,
    n_terms: int = 20,
    score_col: Optional[str] = None,
    ax=None,
):
    """Heatmap / barcode of the top enriched pathways.

    Port of ``OlinkAnalyze::olink_pathway_heatmap``. Plots the top
    ``n_terms`` pathways (by adjusted p) as a horizontal bar of the
    enrichment score (GSEA: NES; ORA: -log10 p).
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * min(n_terms, len(enrichment)) + 1))
    en = enrichment.copy()
    sort_col = "p.adjust" if "p.adjust" in en.columns else "pvalue"
    en = en.sort_values(sort_col).head(n_terms)
    if score_col is None:
        score_col = "NES" if "NES" in en.columns else None
    if score_col and score_col in en.columns:
        vals = en[score_col].to_numpy()
        xlab = score_col
    else:
        vals = -np.log10(en["pvalue"].clip(lower=1e-300).to_numpy())
        xlab = r"$-\log_{10}(p)$"
    y = np.arange(len(en))
    colors = ["#FE1F04" if v >= 0 else "#00559E" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(en["Description"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlab)
    ax.axvline(0, color="0.4", lw=0.6)
    set_plot_theme(ax)
    return ax


def olink_pathway_visualization(
    enrichment: pd.DataFrame,
    n_terms: int = 15,
    ax=None,
):
    """Dot plot of enriched pathways.

    Port of ``OlinkAnalyze::olink_pathway_visualization``. Dot x-axis
    is the enrichment score (NES) or -log10 p; dot size is the set
    size; dot color is the adjusted p-value.
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * min(n_terms, len(enrichment)) + 1))
    en = enrichment.copy()
    sort_col = "p.adjust" if "p.adjust" in en.columns else "pvalue"
    en = en.sort_values(sort_col).head(n_terms)
    if "NES" in en.columns:
        x = en["NES"].to_numpy()
        xlab = "NES"
    else:
        x = -np.log10(en["pvalue"].clip(lower=1e-300).to_numpy())
        xlab = r"$-\log_{10}(p)$"
    y = np.arange(len(en))
    size = en.get("setSize", pd.Series([40] * len(en))).to_numpy()
    padj = en.get("p.adjust", en["pvalue"]).to_numpy()
    sc = ax.scatter(x, y, s=np.clip(size, 10, 300),
                    c=padj, cmap=olink_color_gradient(),
                    edgecolor="0.3", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(en["Description"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlab)
    ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.02,
                       label="adjusted p")
    set_plot_theme(ax)
    return ax
