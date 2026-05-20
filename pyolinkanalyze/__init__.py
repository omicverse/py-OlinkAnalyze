"""
pyolinkanalyze: Pure-Python port of R OlinkAnalyze (Olink Proteomics).

v0.1 surface:

* :func:`read_npx_csv`        — NPX long-format CSV loader.
* :func:`olink_normalization` — bridge / reference-median normalization.
* :func:`olink_ttest`         — Welch two-sample t-test per OlinkID.
* :func:`olink_wilcox`        — Mann-Whitney U per OlinkID.
* :func:`olink_lmer`          — per-protein linear mixed-effects model.
* :func:`olink_volcano_plot`  — minimal volcano helper.
* :func:`olink_qc_plot`       — IQR-based outlier flag plot.

Deferred to v0.2: LOD imputation, pathway enrichment (use ``omicverse.es``),
heatmap / PCA plots, full multi-level Wilcoxon, BAP1-style adjusted
contrasts.
"""
from __future__ import annotations

from .io import read_npx_csv, read_npx_dataframe, write_npx_csv
from .normalization import (
    olink_normalization,
    olink_normalization_reference_medians,
)
from .stats import olink_ttest, olink_wilcox, olink_lmer

try:
    from .plotting import olink_volcano_plot, olink_qc_plot  # noqa: F401
except ImportError:  # matplotlib optional
    olink_volcano_plot = None
    olink_qc_plot = None

__version__ = "0.1.0"

__all__ = [
    "read_npx_csv",
    "read_npx_dataframe",
    "write_npx_csv",
    "olink_normalization",
    "olink_normalization_reference_medians",
    "olink_ttest",
    "olink_wilcox",
    "olink_lmer",
    "olink_volcano_plot",
    "olink_qc_plot",
]
