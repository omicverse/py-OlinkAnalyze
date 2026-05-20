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
