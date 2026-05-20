# pyolinkanalyze

A **pure-Python port of [R OlinkAnalyze](https://cran.r-project.org/package=OlinkAnalyze)** (Olink Proteomics AB) — **comprehensive coverage** of the package's computational API: NPX I/O, bridge / subset / N-way normalization, per-protein differential expression (t-test, Wilcoxon, LMM, ANOVA, Kruskal-Wallis / Friedman, ordinal regression, plus post-hoc contrasts), limit-of-detection handling, plate randomization, pathway enrichment, and a full set of matplotlib plots.

- **No `rpy2`**, no R install. Welch t-test via `scipy.stats.ttest_ind(equal_var=False)`, Mann-Whitney via `scipy.stats.mannwhitneyu(use_continuity=True)`, LMM via `statsmodels.regression.mixed_linear_model.MixedLM`, type-III ANOVA via `statsmodels` + sum-to-zero contrasts, ordinal regression via `statsmodels.miscmodels.ordinal_model.OrderedModel`.
- Tidy long-format `pandas.DataFrame` interface — the same NPX schema Olink ships in their Explore / Target CSVs.
- R-parity tests against `OlinkAnalyze` 3.8.2 — Pearson r > 0.99 (often `=1.0`) on per-protein test statistics and p-values for t-test, Wilcoxon, LMM, ANOVA and Kruskal-Wallis.

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here.

## Install

```bash
pip install pyolinkanalyze
```

Dependencies: `numpy`, `scipy`, `pandas`, `statsmodels`. Plotting needs `matplotlib` + `scikit-learn` (`pip install pyolinkanalyze[plotting]`); `olink_umap_plot` optionally uses `umap-learn` (`pip install pyolinkanalyze[umap]`) and falls back to PCA otherwise.

## Quick-start

```python
import pyolinkanalyze as pa

# Load Olink long-format NPX CSV (auto-detects ; vs , separators)
npx = pa.read_npx_csv("study_NPX_2024.csv")

# Differential expression: two-group Welch t-test per protein
res = pa.olink_ttest(npx, variable="Treatment")
res.head()
# OlinkID  Assay     UniProt  term            estimate  statistic  p.value   Adjusted_pval
# OID00012 IL6       P05231   group1 - group0    1.84    5.12      1.2e-5    8.6e-4
# ...

# Non-parametric alternative
res_w = pa.olink_wilcox(npx, variable="Treatment")

# Linear mixed-effects: NPX ~ Treatment + (1|Subject), per protein
res_lmm = pa.olink_lmer(npx, variable="Treatment", random="Subject")

# Bridge normalization across two batches (4 overlapping samples)
df_ref = pa.read_npx_csv("batch_A.csv")
df_target = pa.read_npx_csv("batch_B.csv")
joined = pa.olink_normalization(
    df_ref, df_target,
    overlapping_samples_df1=["B01", "B02", "B03", "B04"],
    overlapping_samples_df2=["B01", "B02", "B03", "B04"],
)
```

More tests (v0.2):

```python
# Multi-group ANOVA + Tukey post-hoc
res_av = pa.olink_anova(npx, variable="Group")
res_ph = pa.olink_anova_posthoc(npx, variable="Group", effect="Group")

# Non-parametric (Kruskal-Wallis) + Dunn post-hoc
res_kw = pa.olink_one_non_parametric(npx, variable="Group")
res_dunn = pa.olink_one_non_parametric_posthoc(npx, variable="Group")

# Ordinal regression
res_ord = pa.olink_ordinal_regression(npx, variable="Group")

# Limit of detection (negative-control estimate) + below-LOD flags
npx_lod = pa.olink_lod(npx, lod_method="NCLOD")

# Pick optimal bridging samples
bridges = pa.olink_bridge_selector(npx, sample_missing_freq=0.1, n=8)

# Randomize a sample manifest across plates
plated = pa.olink_plate_randomizer(manifest, subject_col="Subject", seed=0)

# Pathway enrichment on a DE result
gene_sets = pa.read_gmt("hallmark.gmt")
enr = pa.olink_pathway_enrichment(res, gene_sets, method="gsea")
```

Plotting helpers:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pa.olink_volcano_plot(res, ax=axes[0])
pa.olink_qc_plot(npx, ax=axes[1])

# v0.2 plots
pa.olink_pca_plot(npx, color_by="Treatment")
pa.olink_heatmap_plot(npx)
pa.olink_boxplot(npx, "Treatment", olinkids=["OID00012"])
pa.olink_pathway_heatmap(enr)
```

## API coverage (v0.2)

Every **computational** function of R OlinkAnalyze is ported.

### I/O & normalization

| Python | R counterpart |
|---|---|
| `read_npx_csv` | `read_NPX` (CSV only — see skipped) |
| `olink_normalization` | `olink_normalization` (bridge, difference-of-medians) |
| `olink_normalization_reference_medians` | `olink_normalization(reference_medians=…)` |
| `olink_normalization_bridge` | `olink_normalization_bridge` (paired median-of-diffs) |
| `olink_normalization_subset` | `olink_normalization_subset` |
| `olink_normalization_n` | `olink_normalization_n` (N-way chain / tree) |
| `olink_bridge_selector` | `olink_bridgeselector` |

### Statistical tests & post-hoc

| Python | R counterpart |
|---|---|
| `olink_ttest` | `olink_ttest` (paired support) |
| `olink_wilcox` | `olink_wilcox` |
| `olink_lmer` | `olink_lmer` |
| `olink_lmer_posthoc` | `olink_lmer_posthoc` (Wald pairwise contrasts) |
| `olink_anova` | `olink_anova` (type-III, `contr.sum`) |
| `olink_anova_posthoc` | `olink_anova_posthoc` (Tukey HSD) |
| `olink_one_non_parametric` | `olink_one_non_parametric` (Kruskal / Friedman) |
| `olink_one_non_parametric_posthoc` | `olink_one_non_parametric_posthoc` (Dunn / paired Wilcoxon) |
| `olink_ordinal_regression` | `olink_ordinalRegression` |
| `olink_ordinal_regression_posthoc` | `olink_ordinalRegression_posthoc` |

### LOD, study design & pathway

| Python | R counterpart |
|---|---|
| `olink_lod` | `olink_lod` (`NCLOD` / `FixedLOD`) |
| `olink_plate_randomizer` | `olink_plate_randomizer` |
| `olink_pathway_enrichment` | `olink_pathway_enrichment` (self-contained GSEA / ORA) |
| `read_gmt` | (helper — load gene sets) |

### Plotting (matplotlib)

| Python | R counterpart |
|---|---|
| `olink_volcano_plot` | `olink_volcano_plot` |
| `olink_qc_plot` | `olink_qc_plot` |
| `olink_boxplot` | `olink_boxplot` |
| `olink_dist_plot` | `olink_dist_plot` |
| `olink_pca_plot` | `olink_pca_plot` (`sklearn.decomposition.PCA`) |
| `olink_umap_plot` | `olink_umap_plot` (`umap-learn`, PCA fallback) |
| `olink_heatmap_plot` | `olink_heatmap_plot` |
| `olink_lmer_plot` | `olink_lmer_plot` |
| `olink_pathway_heatmap` | `olink_pathway_heatmap` |
| `olink_pathway_visualization` | `olink_pathway_visualization` |
| `olink_pal`, `set_plot_theme`, `olink_color_discrete`, `olink_fill_discrete`, `olink_color_gradient`, `olink_fill_gradient` | same names |

## What's NOT ported (and why)

| R function | Reason |
|---|---|
| `olink_displayPlateDistributions`, `olink_displayPlateLayout` | niche plate-layout visualisations — out of scope |
| `read_NPX` (binary / Excel inputs) | `read_npx_csv` covers the long-format CSV path; Olink's binary / Excel parsers are not ported |
| `%>%`, `manifest`, `npx_data1`, `npx_data2` | R pipe + bundled example datasets — not functions |

## R-parity

`tests/test_r_parity.py` (auto-skipped if `OlinkAnalyze` isn't installed in the CMAP R env) compares against `OlinkAnalyze` 3.8.2:

| Quantity | Result |
|---|---|
| `olink_ttest` `estimate` (mean diff) | `atol=1e-8` |
| `olink_ttest` `statistic` / `p.value` | Pearson r > 0.99 |
| `olink_wilcox` `statistic` / `p.value` | `|Pearson r| > 0.99` (R reports `W = U_g1`, scipy reports `U1`) |
| `olink_lmer` F-vs-t² / `p.value` | Pearson r > 0.95 |
| `olink_anova` F-statistic / `p.value` | Pearson **r = 1.0000** (50 proteins) |
| `olink_one_non_parametric` Kruskal stat / `p.value` | Pearson **r = 1.0000** (50 proteins) |
| `olink_bridge_selector` selected sample set | **100 %** overlap with R |
| `olink_lod` below-LOD flags | > 95 % agreement |

## Benchmark

200 proteins × 32 samples, 2 groups:

```bash
python examples/benchmark.py --runs 2
```

Typical Python pipeline wall-time:

| Function | Python (ms) |
|---|---|
| `olink_ttest`  | ~400 |
| `olink_wilcox` | ~255 |

(LMM is dominated by `statsmodels`' per-protein fit — call out `n_jobs` parallelism in v0.2.)

## Notes on the algorithm match

- **t-test**: Welch unequal-variance with the Satterthwaite DF formula. `scipy.stats.ttest_ind(equal_var=False)` matches R `t.test(var.equal=FALSE)` exactly.
- **Wilcoxon**: Asymptotic Mann-Whitney U with Yates continuity correction (`scipy.stats.mannwhitneyu(use_continuity=True, method='asymptotic')`) matches R `wilcox.test(exact=FALSE, correct=TRUE)`. Note R reports `W = U_{g1}` while scipy reports `U_1` for the first sample — Pearson r is essentially `±1` depending on group ordering.
- **LMM**: `statsmodels.mixedlm` fits ML by default (set `reml=False` to match `lme4::lmer(REML=FALSE)`). For REML, pass `reml=True` to the underlying model — fixed-effect coefficients agree at ~1e-5.
- **BH adjustment**: `false_discovery_control(method='bh')` matches `stats::p.adjust(method='BH')` exactly.

## Reproducing R results exactly

```bash
# Requires OlinkAnalyze in the CMAP R env
pytest tests/test_r_parity.py -v
```

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: `omicverse.protein.tl.de(adata, method='ttest', platform='olink')`
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging.

## Citation

If you use this package, please cite the upstream OlinkAnalyze package:

> Olink Proteomics AB. **OlinkAnalyze: Facilitate Analysis of Proteomic Data from Olink.** R package version 5.0.0. https://cran.r-project.org/package=OlinkAnalyze

…and acknowledge omicverse / this repo for the Python port.

## License

AGPL-3.0 — matches the upstream CRAN package.
