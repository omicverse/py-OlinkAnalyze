# pyolinkanalyze

A **pure-Python port of [R OlinkAnalyze](https://cran.r-project.org/package=OlinkAnalyze)** (Olink Proteomics AB) — the core 80% of the package: NPX I/O, bridge normalization, per-protein differential expression (t-test, Wilcoxon, LMM), and minimal volcano / QC plots.

- **No `rpy2`**, no R install. Welch t-test via `scipy.stats.ttest_ind(equal_var=False)`, Mann-Whitney via `scipy.stats.mannwhitneyu(use_continuity=True)`, LMM via `statsmodels.regression.mixed_linear_model.MixedLM`.
- Tidy long-format `pandas.DataFrame` interface — the same NPX schema Olink ships in their Explore / Target CSVs.
- R-parity tests against `OlinkAnalyze::olink_ttest` / `olink_wilcox` — Pearson r > 0.99 on per-protein test statistics and p-values.

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here.

## Install

```bash
pip install pyolinkanalyze
```

Dependencies: `numpy`, `scipy`, `pandas`, `statsmodels`. `matplotlib` is optional (`pip install pyolinkanalyze[plotting]`).

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

Plotting helpers:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pa.olink_volcano_plot(res, ax=axes[0])
pa.olink_qc_plot(npx, ax=axes[1])
```

## API surface (v0.1)

| Python | R counterpart | Status |
|---|---|---|
| `read_npx_csv` | `read_NPX` | v0.1 (auto-detects `,` / `;`) |
| `olink_normalization` (bridge) | `olink_normalization` | v0.1 |
| `olink_normalization_reference_medians` | `olink_normalization(reference_medians=...)` | v0.1 |
| `olink_ttest` | `olink_ttest` | v0.1, paired support |
| `olink_wilcox` | `olink_wilcox` | v0.1 |
| `olink_lmer` | `olink_lmer` | v0.1 (single random group) |
| `olink_volcano_plot` | `olink_volcano_plot` | v0.1 minimal |
| `olink_qc_plot` | `olink_qc_plot` | v0.1 (IQR outlier detection) |

## What's NOT in v0.1

| R function | Status |
|---|---|
| `olink_lod()` LOD imputation | deferred to v0.2 |
| `olink_pathway_enrichment()` | deferred to v0.2 — use `omicverse.es.geneset_enrichment` |
| `olink_heatmap_plot`, `olink_pca_plot`, `olink_boxplot` | deferred to v0.2 — use `omicverse.pl` helpers |
| `olink_anova` multi-level fixed effects | deferred to v0.2 |
| Multi-grouping `olink_lmer` (e.g. crossed random effects) | deferred to v0.3 |
| `olink_normalization_n` (N-way bridging) | deferred to v0.3 |

## R-parity

`tests/test_r_parity.py` (auto-skipped if `OlinkAnalyze` isn't installed in the CMAP R env) compares against `OlinkAnalyze::olink_ttest` / `olink_wilcox`:

| Quantity | Tolerance |
|---|---|
| `olink_ttest` `estimate` (mean diff) | `atol=1e-8` |
| `olink_ttest` `statistic` | Pearson r > 0.99 |
| `olink_ttest` `p.value` | Pearson r > 0.99 |
| `olink_wilcox` `statistic` | `|Pearson r| > 0.99` (R reports `W = U_g1`, scipy reports `U1`) |
| `olink_wilcox` `p.value` | Pearson r > 0.99 |

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
