"""pyolinkanalyze: Pure-Python port of R OlinkAnalyze (Olink Proteomics).

v0.2 surface — comprehensive coverage of the OlinkAnalyze computational
API:

I/O & normalization
    :func:`read_npx`, :func:`read_npx_csv`, :func:`read_npx_excel`,
    :func:`olink_normalization`,
    :func:`olink_normalization_reference_medians`,
    :func:`olink_normalization_bridge`,
    :func:`olink_normalization_subset`,
    :func:`olink_normalization_n`,
    :func:`olink_bridge_selector`.

Statistical tests
    :func:`olink_ttest`, :func:`olink_wilcox`, :func:`olink_lmer`,
    :func:`olink_lmer_posthoc`, :func:`olink_anova`,
    :func:`olink_anova_posthoc`, :func:`olink_one_non_parametric`,
    :func:`olink_one_non_parametric_posthoc`,
    :func:`olink_ordinal_regression`,
    :func:`olink_ordinal_regression_posthoc`.

LOD & study design
    :func:`olink_lod`, :func:`olink_plate_randomizer`.

Pathway enrichment
    :func:`olink_pathway_enrichment`, :func:`read_gmt`.

Plotting (matplotlib)
    :func:`olink_volcano_plot`, :func:`olink_qc_plot`,
    :func:`olink_boxplot`, :func:`olink_dist_plot`,
    :func:`olink_pca_plot`, :func:`olink_umap_plot`,
    :func:`olink_heatmap_plot`, :func:`olink_lmer_plot`,
    :func:`olink_pathway_heatmap`, :func:`olink_pathway_visualization`,
    :func:`olink_display_plate_distributions`,
    :func:`olink_display_plate_layout`,
    :func:`olink_pal`, :func:`set_plot_theme` and the
    ``olink_color_*`` / ``olink_fill_*`` palette helpers.
"""
from __future__ import annotations

from .io import (
    read_npx,
    read_npx_csv,
    read_npx_excel,
    read_npx_dataframe,
    write_npx_csv,
)
from .normalization import (
    olink_normalization,
    olink_normalization_reference_medians,
    olink_normalization_bridge,
    olink_normalization_subset,
    olink_normalization_n,
    olink_bridge_selector,
)
from .stats import (
    olink_ttest,
    olink_wilcox,
    olink_lmer,
    olink_lmer_posthoc,
    olink_anova,
    olink_anova_posthoc,
    olink_one_non_parametric,
    olink_one_non_parametric_posthoc,
    olink_ordinal_regression,
    olink_ordinal_regression_posthoc,
)
from .lod import olink_lod
from .design import olink_plate_randomizer
from .pathway import olink_pathway_enrichment, read_gmt

try:
    from .plotting import (  # noqa: F401
        olink_volcano_plot,
        olink_qc_plot,
        olink_boxplot,
        olink_dist_plot,
        olink_pca_plot,
        olink_umap_plot,
        olink_heatmap_plot,
        olink_lmer_plot,
        olink_pathway_heatmap,
        olink_pathway_visualization,
        olink_display_plate_distributions,
        olink_display_plate_layout,
        olink_pal,
        olink_color_discrete,
        olink_fill_discrete,
        olink_color_gradient,
        olink_fill_gradient,
        set_plot_theme,
    )
except ImportError:  # matplotlib optional
    olink_volcano_plot = None
    olink_qc_plot = None
    olink_boxplot = None
    olink_dist_plot = None
    olink_pca_plot = None
    olink_umap_plot = None
    olink_heatmap_plot = None
    olink_lmer_plot = None
    olink_pathway_heatmap = None
    olink_pathway_visualization = None
    olink_display_plate_distributions = None
    olink_display_plate_layout = None
    olink_pal = None
    olink_color_discrete = None
    olink_fill_discrete = None
    olink_color_gradient = None
    olink_fill_gradient = None
    set_plot_theme = None

__version__ = "0.2.1"

__all__ = [
    # I/O
    "read_npx",
    "read_npx_csv",
    "read_npx_excel",
    "read_npx_dataframe",
    "write_npx_csv",
    # normalization
    "olink_normalization",
    "olink_normalization_reference_medians",
    "olink_normalization_bridge",
    "olink_normalization_subset",
    "olink_normalization_n",
    "olink_bridge_selector",
    # stats
    "olink_ttest",
    "olink_wilcox",
    "olink_lmer",
    "olink_lmer_posthoc",
    "olink_anova",
    "olink_anova_posthoc",
    "olink_one_non_parametric",
    "olink_one_non_parametric_posthoc",
    "olink_ordinal_regression",
    "olink_ordinal_regression_posthoc",
    # LOD & design
    "olink_lod",
    "olink_plate_randomizer",
    # pathway
    "olink_pathway_enrichment",
    "read_gmt",
    # plotting
    "olink_volcano_plot",
    "olink_qc_plot",
    "olink_boxplot",
    "olink_dist_plot",
    "olink_pca_plot",
    "olink_umap_plot",
    "olink_heatmap_plot",
    "olink_lmer_plot",
    "olink_pathway_heatmap",
    "olink_pathway_visualization",
    "olink_display_plate_distributions",
    "olink_display_plate_layout",
    "olink_pal",
    "olink_color_discrete",
    "olink_fill_discrete",
    "olink_color_gradient",
    "olink_fill_gradient",
    "set_plot_theme",
]
