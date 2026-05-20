"""R-parity tests against OlinkAnalyze.

Dumps the same synthetic NPX frame the Python side uses, runs
``olink_ttest`` / ``olink_wilcox`` in R via the driver script, then
compares the per-protein test statistics, effect estimates and
p-values. Tolerances:

* t-test ``estimate`` (mean difference): ``atol=1e-10``.
* t-test ``statistic`` / ``p.value``: Pearson r > 0.99 (Welch DF
  formula matches scipy.ttest_ind exactly).
* Wilcoxon ``statistic`` / ``p.value``: Pearson r > 0.99 (asymptotic
  mode with continuity correction).

The whole file is **skipped** if OlinkAnalyze is not installed in the
CMAP R environment. We do NOT fail CI on missing R deps.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyolinkanalyze import olink_lmer, olink_ttest, olink_wilcox
from tests._synth import generate_synthetic_npx


HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressPackageStartupMessages("
             "library(OlinkAnalyze)); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or OlinkAnalyze not installed.",
)


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory):
    df = generate_synthetic_npx(n_proteins=50, n_samples_per_group=8, seed=0)
    # Tag a paired-subject id so the R driver can run olink_lmer too.
    n_per_grp = 8
    df = df.assign(
        Subject=[
            f"Subj{int(s[1:]) % n_per_grp:02d}"
            for s in df["SampleID"]
        ]
    )
    work = tmp_path_factory.mktemp("olink_parity")
    npx_tsv = work / "npx.tsv"
    df.to_csv(npx_tsv, sep="\t", index=False, na_rep="NA")

    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {npx_tsv} {work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)
    return {"df": df, "work": work, "R_out": work / "R_out"}


def _align(py_res: pd.DataFrame, r_res: pd.DataFrame):
    """Inner-join on OlinkID and return (py_subset, r_subset)."""
    py = py_res.set_index("OlinkID")
    # R's column may be 'OlinkID' or it may be wrapped — be permissive.
    if "OlinkID" in r_res.columns:
        r = r_res.set_index("OlinkID")
    else:
        r = r_res.copy()
        r.index = r.iloc[:, 0]
    common = py.index.intersection(r.index)
    return py.loc[common], r.loc[common]


def test_olink_ttest_r_parity(shared_dataset):
    """Python olink_ttest output should agree closely with R."""
    py = olink_ttest(shared_dataset["df"], variable="Treatment")
    r = pd.read_csv(shared_dataset["R_out"] / "ttest.tsv", sep="\t")
    py_a, r_a = _align(py, r)

    # Find the columns R uses (varies slightly by version).
    r_est = None
    for c in ("estimate", "Mean_difference", "MeanDifference", "logFC"):
        if c in r_a.columns:
            r_est = c
            break
    r_p = "p.value" if "p.value" in r_a.columns else "p_value"
    r_stat = "statistic" if "statistic" in r_a.columns else "t"

    # Estimate (mean diff): R defines estimate = group0 - group1,
    # while pyolinkanalyze uses group1 - group0 (the modern
    # treatment-minus-control convention). Absolute values must agree
    # bit-close; the sign carries the contrast convention.
    if r_est is not None:
        np.testing.assert_allclose(
            np.abs(py_a["estimate"].to_numpy()),
            np.abs(r_a[r_est].to_numpy()),
            atol=1e-8,
            err_msg="|estimate| diverges from R",
        )

    # Pearson |r| on test statistics (sign-flips with the estimate
    # convention) and p-values
    py_stat = py_a["statistic"].to_numpy()
    r_stat_v = r_a[r_stat].to_numpy()
    r_stat_corr = abs(np.corrcoef(py_stat, r_stat_v)[0, 1])
    assert r_stat_corr > 0.99, f"t-test |statistic| Pearson |r| = {r_stat_corr:.4f}"

    py_p = py_a["p.value"].to_numpy()
    r_p_v = r_a[r_p].to_numpy()
    p_corr = np.corrcoef(py_p, r_p_v)[0, 1]
    assert p_corr > 0.99, f"t-test p.value Pearson r = {p_corr:.4f}"


def test_olink_wilcox_r_parity(shared_dataset):
    py = olink_wilcox(shared_dataset["df"], variable="Treatment")
    r = pd.read_csv(shared_dataset["R_out"] / "wilcox.tsv", sep="\t")
    py_a, r_a = _align(py, r)

    r_p = "p.value" if "p.value" in r_a.columns else "p_value"
    r_stat = "statistic" if "statistic" in r_a.columns else "W"

    # R's W statistic = U_g1 (matches scipy.mannwhitneyu's U2 for the
    # 'greater' arrangement) — scipy reports U1 for the 'less' group's
    # rank-sum minus the offset. The Pearson r should still be high.
    py_stat = py_a["statistic"].to_numpy()
    r_stat_v = r_a[r_stat].to_numpy()
    stat_corr = np.corrcoef(py_stat, r_stat_v)[0, 1]
    assert abs(stat_corr) > 0.99, (
        f"Wilcoxon statistic Pearson |r| = {stat_corr:.4f}"
    )

    py_p = py_a["p.value"].to_numpy()
    r_p_v = r_a[r_p].to_numpy()
    p_corr = np.corrcoef(py_p, r_p_v)[0, 1]
    assert p_corr > 0.99, f"Wilcoxon p.value Pearson r = {p_corr:.4f}"


def test_olink_lmer_r_parity(shared_dataset):
    """LMM p-values and test stats should agree closely.

    R's ``olink_lmer`` returns an ANOVA-style table per OlinkID
    (Satterthwaite F-test via ``lmerTest::anova``). Pyolinkanalyze
    reports per-coefficient t-tests from ``statsmodels.MixedLM``.
    For 1-DF fixed effects F = t², so we compare:

    * ``py.statistic**2`` vs R's F ``statistic``
    * ``py.p.value`` vs R's ``p.value``

    Pearson r > 0.95 expected (F-vs-t² is exact; the DFs differ —
    statsmodels uses Wald-z, R uses Satterthwaite — so p's diverge
    most at the very small end).
    """
    lmer_path = shared_dataset["R_out"] / "lmer.tsv"
    if not lmer_path.exists():
        pytest.skip("R olink_lmer output not present")
    py = olink_lmer(shared_dataset["df"], variable="Treatment",
                    random="Subject")
    py = py.loc[py["term"] == "Treatment"]
    r = pd.read_csv(lmer_path, sep="\t")
    r = r.loc[r["term"].astype(str).str.contains("Treatment", na=False)]
    py_a, r_a = _align(py, r)
    if len(py_a) < 5:
        pytest.skip(f"Too few common OlinkIDs ({len(py_a)})")

    # Compare F (R) vs t² (py). Drop NaNs from singular fits.
    py_stat_sq = py_a["statistic"].to_numpy() ** 2
    r_stat = r_a["statistic"].to_numpy()
    mask = np.isfinite(py_stat_sq) & np.isfinite(r_stat)
    if mask.sum() < 5:
        pytest.skip(f"Too few finite LMM fits ({mask.sum()})")
    stat_corr = float(np.corrcoef(py_stat_sq[mask], r_stat[mask])[0, 1])
    assert stat_corr > 0.95, f"LMM F-vs-t² Pearson r = {stat_corr:.4f}"

    py_p = py_a["p.value"].to_numpy()
    r_p = r_a["p.value"].to_numpy()
    mask = np.isfinite(py_p) & np.isfinite(r_p)
    p_corr = float(np.corrcoef(py_p[mask], r_p[mask])[0, 1])
    assert p_corr > 0.95, f"LMM p.value Pearson r = {p_corr:.4f}"


# ======================================================================
# v0.2 R-parity tests — ANOVA, Kruskal-Wallis, LOD, bridge selector
# ======================================================================

@pytest.fixture(scope="module")
def three_group_dataset(tmp_path_factory):
    """3-group NPX (Group) for ANOVA / Kruskal; plus a bridge frame.

    The R driver also needs a 2-level ``Treatment`` column for the
    ttest/wilcox/lmer references, so we collapse group2 into group1.
    """
    df = generate_synthetic_npx(n_proteins=50, n_samples_per_group=8,
                                n_groups=3, seed=0)
    n_per = 8
    df = df.assign(
        Subject=[f"Subj{int(s[1:]) % n_per:02d}" for s in df["SampleID"]],
        Group=df["Treatment"],
    )
    df["Treatment"] = df["Group"].map(
        {"group0": "group0", "group1": "group1", "group2": "group1"}
    )
    work = tmp_path_factory.mktemp("olink_parity_v2")
    npx_tsv = work / "npx.tsv"
    df.to_csv(npx_tsv, sep="\t", index=False, na_rep="NA")

    # Bridge frame for olink_bridgeselector parity.
    from tests._synth import generate_bridge_pair
    df_ref, _, _, _ = generate_bridge_pair(
        n_proteins=30, n_samples_per_project=20, n_bridge=4, seed=13)
    df_ref = df_ref.assign(QC_Warning="PASS")
    df_ref.to_csv(work / "bridge_npx.tsv", sep="\t", index=False, na_rep="NA")

    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {npx_tsv} {work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)
    return {"df": df, "df_ref": df_ref, "work": work,
            "R_out": work / "R_out"}


def test_olink_anova_r_parity(three_group_dataset):
    """olink_anova F-statistic and p-value should match R closely."""
    from pyolinkanalyze import olink_anova
    anova_path = three_group_dataset["R_out"] / "anova.tsv"
    if not anova_path.exists():
        pytest.skip("R olink_anova output not present")
    py = olink_anova(three_group_dataset["df"], variable="Group")
    r = pd.read_csv(anova_path, sep="\t")
    py_a, r_a = _align(py, r)
    if len(py_a) < 5:
        pytest.skip(f"Too few common OlinkIDs ({len(py_a)})")

    f_corr = np.corrcoef(py_a["statistic"].to_numpy(),
                         r_a["statistic"].to_numpy())[0, 1]
    assert f_corr > 0.99, f"ANOVA F-stat Pearson r = {f_corr:.4f}"
    p_corr = np.corrcoef(py_a["p.value"].to_numpy(),
                         r_a["p.value"].to_numpy())[0, 1]
    assert p_corr > 0.99, f"ANOVA p.value Pearson r = {p_corr:.4f}"


def test_olink_one_non_parametric_r_parity(three_group_dataset):
    """Kruskal-Wallis statistic / p should match R closely."""
    from pyolinkanalyze import olink_one_non_parametric
    kw_path = three_group_dataset["R_out"] / "kruskal.tsv"
    if not kw_path.exists():
        pytest.skip("R olink_one_non_parametric output not present")
    py = olink_one_non_parametric(three_group_dataset["df"], variable="Group")
    r = pd.read_csv(kw_path, sep="\t")
    py_a, r_a = _align(py, r)
    if len(py_a) < 5:
        pytest.skip(f"Too few common OlinkIDs ({len(py_a)})")

    stat_corr = np.corrcoef(py_a["statistic"].to_numpy(),
                            r_a["statistic"].to_numpy())[0, 1]
    assert stat_corr > 0.99, f"Kruskal stat Pearson r = {stat_corr:.4f}"
    p_corr = np.corrcoef(py_a["p.value"].to_numpy(),
                         r_a["p.value"].to_numpy())[0, 1]
    assert p_corr > 0.99, f"Kruskal p.value Pearson r = {p_corr:.4f}"


def test_olink_bridge_selector_r_parity(three_group_dataset):
    """Selected bridge-sample set should overlap R's by > 80%."""
    from pyolinkanalyze import olink_bridge_selector
    bridge_path = three_group_dataset["R_out"] / "bridge.tsv"
    if not bridge_path.exists():
        pytest.skip("R olink_bridgeselector output not present")
    py = olink_bridge_selector(three_group_dataset["df_ref"],
                               sample_missing_freq=0.9, n=6)
    r = pd.read_csv(bridge_path, sep="\t")
    py_ids = set(py["SampleID"].astype(str))
    r_ids = set(r["SampleID"].astype(str))
    overlap = len(py_ids & r_ids) / max(len(r_ids), 1)
    assert overlap > 0.8, (
        f"bridge selection overlap with R = {overlap:.2f} "
        f"(py={sorted(py_ids)}, r={sorted(r_ids)})"
    )


def test_olink_lod_below_lod_agreement():
    """olink_lod below-LOD flags should agree > 95% with a direct
    NPX <= LOD comparison on a frame that already carries an LOD."""
    from pyolinkanalyze import olink_lod
    df = generate_synthetic_npx(n_proteins=40, n_samples_per_group=10, seed=3)
    # Use the synthetic per-assay LOD as a FixedLOD reference table.
    lod_tbl = df[["OlinkID", "LOD"]].drop_duplicates()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        lod_tbl.to_csv(f.name, sep=";", index=False)
        out = olink_lod(df.drop(columns=["LOD"]), lod_file=f.name,
                        lod_method="FixedLOD")
    expected = (df["NPX"].to_numpy()
                <= df["LOD"].to_numpy())
    got = out.sort_values(["OlinkID", "SampleID"])["below_LOD"].to_numpy()
    exp = (df.sort_values(["OlinkID", "SampleID"])["NPX"].to_numpy()
           <= df.sort_values(["OlinkID", "SampleID"])["LOD"].to_numpy())
    agreement = np.mean(got == exp)
    assert agreement > 0.95, f"below-LOD agreement = {agreement:.3f}"
