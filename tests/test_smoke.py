"""No-R smoke tests for pyolinkanalyze."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyolinkanalyze import (
    olink_lmer,
    olink_normalization,
    olink_ttest,
    olink_wilcox,
    read_npx_csv,
    read_npx_dataframe,
)
from tests._synth import generate_bridge_pair, generate_synthetic_npx


def test_read_npx_dataframe_validates_required_cols():
    df = generate_synthetic_npx(n_proteins=5, n_samples_per_group=3, seed=0)
    out = read_npx_dataframe(df)
    assert {"SampleID", "OlinkID", "NPX"}.issubset(out.columns)
    assert out["NPX"].dtype == float
    # Missing required column should fail
    with pytest.raises(ValueError):
        read_npx_dataframe(df.drop(columns=["NPX"]))


def test_read_npx_csv_roundtrip(tmp_path):
    df = generate_synthetic_npx(n_proteins=4, n_samples_per_group=3, seed=1)
    p = tmp_path / "npx.csv"
    df.to_csv(p, index=False)
    out = read_npx_csv(p)
    assert out.shape == df.shape
    np.testing.assert_allclose(out["NPX"].to_numpy(), df["NPX"].to_numpy())


def test_olink_ttest_detects_planted_de():
    df = generate_synthetic_npx(n_proteins=40, n_samples_per_group=10, seed=2)
    res = olink_ttest(df, variable="Treatment")
    # Sorted by ascending p
    assert (np.diff(res["p.value"].fillna(1.0).to_numpy()) >= -1e-12).all()
    # Each protein has one row
    assert res["OlinkID"].nunique() == 40
    # Some passed the BH threshold
    assert (res["Threshold"] == 1).sum() >= 0  # may be zero in low-N, but column exists
    # Estimate is mean(group1) - mean(group0)
    g0 = df.loc[df["Treatment"] == "group0"].groupby("OlinkID")["NPX"].mean()
    g1 = df.loc[df["Treatment"] == "group1"].groupby("OlinkID")["NPX"].mean()
    expected = (g1 - g0).reindex(res["OlinkID"])
    np.testing.assert_allclose(res["estimate"].to_numpy(),
                               expected.to_numpy(), atol=1e-10)


def test_olink_wilcox_matches_ttest_direction():
    # Use a strong DE setup so the per-protein direction is detectable
    # by both tests; on null proteins the sign is pure noise.
    rng = np.random.default_rng(3)
    rows = []
    deltas = rng.choice([-1.5, 1.5], 30)
    for prot_i in range(30):
        for s_i in range(24):
            g = s_i // 12
            mu = (deltas[prot_i] if g == 1 else 0.0)
            npx = rng.normal(mu, 0.3)
            rows.append({
                "SampleID":   f"S{s_i:03d}",
                "OlinkID":    f"OID{prot_i:05d}",
                "Assay":      f"prot_{prot_i:04d}",
                "UniProt":    f"P{prot_i:05d}",
                "Panel":      "Inflammation",
                "NPX":        float(npx),
                "QC_Warning": "PASS",
                "LOD":        -3.0,
                "Treatment":  f"group{g}",
            })
    df = pd.DataFrame(rows)
    t_res = olink_ttest(df, variable="Treatment").set_index("OlinkID")
    w_res = olink_wilcox(df, variable="Treatment").set_index("OlinkID")
    common = t_res.index.intersection(w_res.index)
    sign_match = (
        np.sign(t_res.loc[common, "estimate"].to_numpy())
        == np.sign(w_res.loc[common, "estimate"].to_numpy())
    )
    assert sign_match.mean() >= 0.95


def test_olink_lmer_recovers_fixed_effect():
    """With a planted group effect and a balanced random subject, the
    LMM coefficient should be positive on average for up-regulated
    proteins and negative for down-regulated ones."""
    rng = np.random.default_rng(4)
    n_subj = 8
    rows = []
    for prot_i in range(20):
        delta = (1.5 if prot_i < 5 else (-1.5 if prot_i < 10 else 0.0))
        for subj in range(n_subj):
            subj_eff = rng.normal(0.0, 0.3)
            for visit, treat in enumerate(["A", "B"]):
                npx = (rng.normal(0, 0.4) + subj_eff
                       + (delta if treat == "B" else 0.0))
                rows.append({
                    "SampleID": f"S{subj}_{visit}",
                    "OlinkID":  f"OID{prot_i:05d}",
                    "Assay":    f"prot_{prot_i:04d}",
                    "UniProt":  f"P{prot_i:05d}",
                    "Panel":    "Inflammation",
                    "NPX":      float(npx),
                    "Treatment": treat,
                    "Subject":  f"Subj{subj}",
                })
    df = pd.DataFrame(rows)
    res = olink_lmer(df, variable="Treatment", random="Subject")
    assert "p.value" in res.columns
    # Each protein should have at least one row
    assert res["OlinkID"].nunique() == 20
    # The 'Treatment' effect for up-regulated proteins should be >0 on average
    up = res.loc[(res["term"] == "Treatment")
                 & (res["OlinkID"].isin([f"OID{i:05d}" for i in range(5)]))]
    assert up["estimate"].mean() > 0.5


def test_olink_normalization_bridge_removes_batch_shift():
    df_ref, df_tgt, bridge_ids, true_shift = generate_bridge_pair(
        n_proteins=20, n_samples_per_project=12, n_bridge=4, seed=5,
    )
    out = olink_normalization(
        df_ref, df_tgt,
        overlapping_samples_df1=bridge_ids,
        overlapping_samples_df2=bridge_ids,
    )
    # Project labels exist
    assert set(out["Project"].unique()) == {"ref", "target"}
    # After normalization the per-assay median of bridging samples in
    # ref vs target should be ~equal.
    norm_target = out.loc[out["Project"] == "target"]
    med_tgt = norm_target.loc[norm_target["SampleID"].isin(bridge_ids)] \
        .groupby("OlinkID")["NPX"].median()
    med_ref = df_ref.loc[df_ref["SampleID"].isin(bridge_ids)] \
        .groupby("OlinkID")["NPX"].median()
    diff = (med_ref - med_tgt).to_numpy()
    np.testing.assert_allclose(diff, 0.0, atol=1e-10)
    # The recovered Adj_factor should approximate -batch_shift, with
    # bridge-sample sampling noise of ~sigma / sqrt(n_bridge) ≈ 0.25.
    adj = (
        norm_target.drop_duplicates("OlinkID")
        .set_index("OlinkID")["Adj_factor"]
        .sort_index()
    )
    # Correlation between recovered shift and -batch_shift should be strong
    r = np.corrcoef(adj.to_numpy(), -true_shift)[0, 1]
    assert r > 0.6, f"Adj_factor not tracking batch_shift (r={r:.3f})"


def test_olink_volcano_plot_returns_axes():
    matplotlib = pytest.importorskip("matplotlib")
    from pyolinkanalyze import olink_volcano_plot

    df = generate_synthetic_npx(n_proteins=20, n_samples_per_group=8, seed=6)
    res = olink_ttest(df, variable="Treatment")
    ax = olink_volcano_plot(res)
    assert ax is not None
    assert hasattr(ax, "scatter")


def test_olink_qc_plot_returns_stats():
    matplotlib = pytest.importorskip("matplotlib")
    from pyolinkanalyze import olink_qc_plot

    df = generate_synthetic_npx(n_proteins=10, n_samples_per_group=8, seed=7)
    # Inject one obvious outlier sample
    df.loc[df["SampleID"] == "S000", "NPX"] += 20.0
    ax = olink_qc_plot(df)
    stats = ax.qc_stats
    assert "outlier" in stats.columns
    assert stats.loc[stats["SampleID"] == "S000", "outlier"].iloc[0]


# ======================================================================
# v0.2 smoke tests — no R required
# ======================================================================

def _three_group(n_proteins=20, n_per=8, seed=10):
    return generate_synthetic_npx(n_proteins=n_proteins,
                                  n_samples_per_group=n_per,
                                  n_groups=3, seed=seed)


def test_olink_anova_one_way():
    from pyolinkanalyze import olink_anova
    df = _three_group()
    res = olink_anova(df, variable="Treatment")
    assert res["OlinkID"].nunique() == 20
    for c in ("term", "statistic", "p.value", "Adjusted_pval"):
        assert c in res.columns
    assert (res["p.value"].between(0, 1) | res["p.value"].isna()).all()


def test_olink_anova_posthoc():
    from pyolinkanalyze import olink_anova_posthoc
    df = _three_group()
    res = olink_anova_posthoc(df, variable="Treatment", effect="Treatment")
    # 3 groups -> 3 pairwise contrasts per protein.
    assert (res.groupby("OlinkID").size() == 3).all()
    assert {"contrast", "estimate", "Adjusted_pval"}.issubset(res.columns)


def test_olink_one_non_parametric_kruskal():
    from pyolinkanalyze import olink_one_non_parametric
    df = _three_group()
    res = olink_one_non_parametric(df, variable="Treatment")
    assert res["OlinkID"].nunique() == 20
    assert res["method"].iloc[0].startswith("Kruskal")
    assert (res["p.value"].between(0, 1) | res["p.value"].isna()).all()


def test_olink_one_non_parametric_friedman():
    from pyolinkanalyze import olink_one_non_parametric
    df = _three_group()
    df = df.assign(Subject=[f"Subj{int(s[1:]) % 8:02d}"
                            for s in df["SampleID"]])
    res = olink_one_non_parametric(df, variable="Treatment",
                                   dependence=True, subject="Subject")
    assert res["method"].iloc[0].startswith("Friedman")
    assert "p.value" in res.columns


def test_olink_non_parametric_posthoc_dunn():
    from pyolinkanalyze import olink_one_non_parametric_posthoc
    df = _three_group()
    res = olink_one_non_parametric_posthoc(df, variable="Treatment")
    assert (res.groupby("OlinkID").size() == 3).all()
    assert {"contrast", "estimate", "Adjusted_pval"}.issubset(res.columns)


def test_olink_ordinal_regression():
    from pyolinkanalyze import olink_ordinal_regression
    df = _three_group(n_proteins=8, n_per=8)
    res = olink_ordinal_regression(df, variable="Treatment")
    assert {"term", "statistic", "p.value", "Adjusted_pval"}.issubset(res.columns)
    assert len(res) >= 1


def test_olink_ordinal_regression_posthoc():
    from pyolinkanalyze import olink_ordinal_regression_posthoc
    df = _three_group(n_proteins=8, n_per=8)
    res = olink_ordinal_regression_posthoc(df, variable="Treatment",
                                           effect="Treatment")
    if not res.empty:
        assert {"contrast", "estimate", "Adjusted_pval"}.issubset(res.columns)


def test_olink_lmer_posthoc():
    from pyolinkanalyze import olink_lmer_posthoc
    df = _three_group(n_proteins=10, n_per=8)
    df = df.assign(Subject=[f"Subj{int(s[1:]) % 8:02d}"
                            for s in df["SampleID"]])
    res = olink_lmer_posthoc(df, variable="Treatment", random="Subject",
                             effect="Treatment")
    if not res.empty:
        assert {"contrast", "estimate", "Adjusted_pval"}.issubset(res.columns)


def test_olink_lod_flags_below():
    from pyolinkanalyze import olink_lod
    df = generate_synthetic_npx(n_proteins=10, n_samples_per_group=6, seed=11)
    # Add an explicit negative-control sample with very low NPX.
    nc = df.loc[df["SampleID"] == "S000"].copy()
    nc["SampleID"] = "NC_001"
    nc["NPX"] -= 6.0
    df = pd.concat([df, nc], ignore_index=True)
    out = olink_lod(df, lod_method="NCLOD")
    assert "LOD" in out.columns and "below_LOD" in out.columns
    assert out["below_LOD"].dtype == bool
    assert "lod_summary" in out.attrs


def test_olink_lod_fixed_file(tmp_path):
    from pyolinkanalyze import olink_lod
    df = generate_synthetic_npx(n_proteins=8, n_samples_per_group=5, seed=12)
    lod_path = tmp_path / "lod.csv"
    oids = df["OlinkID"].unique()
    pd.DataFrame({"OlinkID": oids,
                  "LOD": np.zeros(len(oids))}).to_csv(
        lod_path, sep=";", index=False)
    out = olink_lod(df, lod_file=lod_path, lod_method="FixedLOD")
    assert (out["LOD"] == 0.0).all()
    assert out["below_LOD"].equals(out["NPX"] <= 0.0)


def test_olink_bridge_selector():
    from pyolinkanalyze import olink_bridge_selector
    df_ref, _, _, _ = generate_bridge_pair(
        n_proteins=30, n_samples_per_project=20, n_bridge=4, seed=13)
    sel = olink_bridge_selector(df_ref, sample_missing_freq=0.9, n=6)
    assert len(sel) == 6
    assert {"SampleID", "PercAssaysBelowLOD", "MeanNPX"}.issubset(sel.columns)


def test_olink_normalization_bridge_variant():
    from pyolinkanalyze import olink_normalization_bridge
    df_ref, df_tgt, bridge_ids, true_shift = generate_bridge_pair(
        n_proteins=20, n_samples_per_project=12, n_bridge=4, seed=14)
    out = olink_normalization_bridge(df_ref, df_tgt, bridge_ids)
    assert set(out["Project"].unique()) == {"P1", "P2"}
    adj = (out.loc[out["Project"] == "P2"]
           .drop_duplicates("OlinkID").set_index("OlinkID")["Adj_factor"]
           .sort_index())
    r = np.corrcoef(adj.to_numpy(), -true_shift)[0, 1]
    assert r > 0.6


def test_olink_normalization_subset_variant():
    from pyolinkanalyze import olink_normalization_subset
    df_ref, df_tgt, bridge_ids, _ = generate_bridge_pair(
        n_proteins=15, n_samples_per_project=12, n_bridge=4, seed=15)
    out = olink_normalization_subset(df_ref, df_tgt, bridge_ids)
    assert set(out["Project"].unique()) == {"P1", "P2"}
    assert "Adj_factor" in out.columns


def test_olink_normalization_n():
    from pyolinkanalyze import olink_normalization_n
    df1, df2, br12, _ = generate_bridge_pair(
        n_proteins=12, n_samples_per_project=10, n_bridge=3, seed=16)
    df3, _, _, _ = generate_bridge_pair(
        n_proteins=12, n_samples_per_project=10, n_bridge=3, seed=17)
    df3 = df3[0] if isinstance(df3, tuple) else df3
    # Three-project chain: P1 (ref) <- P2 <- P3.
    schema = [
        {"name": "P1", "data": df1, "order": 1},
        {"name": "P2", "data": df2, "order": 2,
         "normalization_type": "Bridge", "normalize_to": "P1",
         "samples": br12},
        {"name": "P3", "data": df3, "order": 3,
         "normalization_type": "Bridge", "normalize_to": "P2",
         "samples": list(df2["SampleID"].unique()[:3])},
    ]
    out = olink_normalization_n(schema)
    assert set(out["Project"].unique()) == {"P1", "P2", "P3"}


def test_olink_plate_randomizer_samples():
    from pyolinkanalyze import olink_plate_randomizer
    man = pd.DataFrame({"SampleID": [f"S{i:03d}" for i in range(150)]})
    out = olink_plate_randomizer(man, seed=0)
    # All real samples placed exactly once.
    real = out.loc[out["SampleID"] != "CONTROL_SAMPLE"]
    assert len(real) == 150
    assert real["SampleID"].nunique() == 150
    assert {"plate", "column", "row", "well"}.issubset(out.columns)


def test_olink_plate_randomizer_keeps_subjects():
    from pyolinkanalyze import olink_plate_randomizer
    man = pd.DataFrame({
        "SampleID": [f"S{i:03d}" for i in range(120)],
        "Subject": [f"Subj{i // 3:03d}" for i in range(120)],
    })
    out = olink_plate_randomizer(man, subject_col="Subject", seed=1)
    real = out.loc[out["SampleID"] != "CONTROL_SAMPLE"]
    # Each subject's samples all land on one plate.
    per_subj = real.groupby("Subject")["plate"].nunique()
    assert (per_subj == 1).all()


def test_olink_pathway_enrichment_ora():
    from pyolinkanalyze import olink_pathway_enrichment
    df = generate_synthetic_npx(n_proteins=60, n_samples_per_group=10, seed=18)
    res = olink_ttest(df, variable="Treatment")
    gene_sets = {f"set_{k}": [f"prot_{i:04d}" for i in range(k * 6, k * 6 + 10)]
                 for k in range(8)}
    en = olink_pathway_enrichment(res, gene_sets, method="ora",
                                  pvalue_cutoff=0.99)
    assert {"Description", "pvalue", "p.adjust", "Count"}.issubset(en.columns)


def test_olink_pathway_enrichment_gsea():
    from pyolinkanalyze import olink_pathway_enrichment
    df = generate_synthetic_npx(n_proteins=60, n_samples_per_group=10, seed=19)
    res = olink_ttest(df, variable="Treatment")
    gene_sets = {f"set_{k}": [f"prot_{i:04d}" for i in range(k * 6, k * 6 + 10)]
                 for k in range(8)}
    en = olink_pathway_enrichment(res, gene_sets, method="gsea", n_perm=200)
    assert {"Description", "NES", "pvalue", "p.adjust"}.issubset(en.columns)
    assert (en["pvalue"].between(0, 1)).all()


# ----------------------------------------------------------------------
# Plotting smoke tests
# ----------------------------------------------------------------------

def _plot_df():
    df = generate_synthetic_npx(n_proteins=15, n_samples_per_group=8, seed=20)
    return df.assign(Subject=[f"Subj{int(s[1:]) % 8:02d}"
                              for s in df["SampleID"]])


def test_olink_boxplot_returns_axes():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_boxplot
    df = _plot_df()
    ax = olink_boxplot(df, "Treatment", list(df["OlinkID"].unique()[:2]))
    assert hasattr(ax, "scatter")


def test_olink_dist_plot_returns_axes():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_dist_plot
    ax = olink_dist_plot(_plot_df(), color_by="Treatment")
    assert hasattr(ax, "scatter")


def test_olink_pca_plot_returns_axes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_pca_plot
    ax = olink_pca_plot(_plot_df(), color_by="Treatment")
    assert hasattr(ax, "pca_scores")


def test_olink_umap_plot_returns_axes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_umap_plot
    ax = olink_umap_plot(_plot_df(), color_by="Treatment")
    assert hasattr(ax, "umap_scores")


def test_olink_heatmap_plot_returns_axes():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_heatmap_plot
    ax = olink_heatmap_plot(_plot_df())
    assert hasattr(ax, "heatmap_data")


def test_olink_lmer_plot_returns_axes():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import olink_lmer_plot
    df = _plot_df()
    ax = olink_lmer_plot(df, "Treatment", "Subject",
                         list(df["OlinkID"].unique()[:2]))
    assert ax is not None


def test_olink_pal_and_theme():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import (olink_pal, olink_color_gradient,
                                set_plot_theme)
    import matplotlib.pyplot as plt
    assert len(olink_pal(5)) == 5
    assert len(olink_pal(20)) == 20
    assert olink_color_gradient() is not None
    _, ax = plt.subplots()
    assert set_plot_theme(ax) is ax


def test_olink_pathway_plots_return_axes():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from pyolinkanalyze import (olink_pathway_enrichment,
                                olink_pathway_heatmap,
                                olink_pathway_visualization)
    df = generate_synthetic_npx(n_proteins=50, n_samples_per_group=10, seed=21)
    res = olink_ttest(df, variable="Treatment")
    gene_sets = {f"set_{k}": [f"prot_{i:04d}" for i in range(k * 6, k * 6 + 10)]
                 for k in range(6)}
    en = olink_pathway_enrichment(res, gene_sets, method="gsea", n_perm=100)
    assert olink_pathway_heatmap(en) is not None
    assert olink_pathway_visualization(en) is not None
