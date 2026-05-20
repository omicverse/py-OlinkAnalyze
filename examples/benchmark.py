"""Head-to-head benchmark: R OlinkAnalyze vs pyolinkanalyze.

Runs ``olink_ttest`` and ``olink_wilcox`` on a synthetic NPX dataset
(200 proteins × 32 samples × 2 groups) and reports wall-clock per
function plus per-protein agreement (Pearson r on p-values).

Auto-skips the R half if OlinkAnalyze isn't installed in the CMAP
environment.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make tests/_synth importable from a sibling examples/ dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyolinkanalyze import olink_ttest, olink_wilcox  # noqa: E402
from tests._synth import generate_synthetic_npx  # noqa: E402


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = ROOT / "tests" / "r_reference_driver.R"


def _r_available() -> bool:
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} && Rscript -e "
             "'suppressPackageStartupMessages(library(OlinkAnalyze));cat(\"OK\")'"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


def run_python(df, runs):
    elapsed_t, elapsed_w = [], []
    res_t = res_w = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res_t = olink_ttest(df, variable="Treatment")
        elapsed_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        res_w = olink_wilcox(df, variable="Treatment")
        elapsed_w.append(time.perf_counter() - t0)
    return (float(np.mean(elapsed_t)), float(np.mean(elapsed_w)),
            res_t, res_w)


def run_R(work):
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && Rscript "
        f"{R_DRIVER} {work / 'npx.tsv'} {work / 'R_out'}"
    )
    t0 = time.perf_counter()
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    timing = pd.read_csv(work / "R_out" / "timing.tsv", sep="\t")
    return elapsed, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-proteins", type=int, default=200)
    ap.add_argument("--n-per-group", type=int, default=16)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    df = generate_synthetic_npx(
        n_proteins=args.n_proteins,
        n_samples_per_group=args.n_per_group,
        seed=42,
    )
    df.to_csv(WORK / "npx.tsv", sep="\t", index=False, na_rep="NA")
    print(f"NPX: {df['OlinkID'].nunique()} proteins × "
          f"{df['SampleID'].nunique()} samples")

    py_t, py_w, py_res_t, py_res_w = run_python(df, args.runs)
    print(f"\n--- Python timing (mean of {args.runs} runs) ---")
    print(f"  olink_ttest        {py_t*1000:8.2f} ms")
    print(f"  olink_wilcox       {py_w*1000:8.2f} ms")

    summary = {
        "shape": [df["OlinkID"].nunique(), df["SampleID"].nunique()],
        "py_ttest_ms": py_t * 1000,
        "py_wilcox_ms": py_w * 1000,
    }

    if _r_available():
        r_total, r_timing = run_R(WORK)
        print(f"\n--- R timing (single run; per-function breakdown) ---")
        print(r_timing.to_string(index=False))
        print(f"  R total wall       {r_total*1000:8.2f} ms")

        r_t = float(r_timing["ttest"].iloc[0])
        r_w = float(r_timing["wilcox"].iloc[0])
        print(f"\nSpeedup (R / Python):")
        print(f"  olink_ttest  {r_t/py_t:.2f}×")
        print(f"  olink_wilcox {r_w/py_w:.2f}×")

        # Accuracy
        r_tt = pd.read_csv(WORK / "R_out" / "ttest.tsv", sep="\t")
        r_ww = pd.read_csv(WORK / "R_out" / "wilcox.tsv", sep="\t")
        py_t_idx = py_res_t.set_index("OlinkID")
        py_w_idx = py_res_w.set_index("OlinkID")
        if "OlinkID" in r_tt.columns:
            r_tt = r_tt.set_index("OlinkID")
        if "OlinkID" in r_ww.columns:
            r_ww = r_ww.set_index("OlinkID")
        common_t = py_t_idx.index.intersection(r_tt.index)
        common_w = py_w_idx.index.intersection(r_ww.index)

        r_p_col_t = "p.value" if "p.value" in r_tt.columns else "p_value"
        r_p_col_w = "p.value" if "p.value" in r_ww.columns else "p_value"
        tt_corr = float(np.corrcoef(
            py_t_idx.loc[common_t, "p.value"].to_numpy(),
            r_tt.loc[common_t, r_p_col_t].to_numpy())[0, 1])
        ww_corr = float(np.corrcoef(
            py_w_idx.loc[common_w, "p.value"].to_numpy(),
            r_ww.loc[common_w, r_p_col_w].to_numpy())[0, 1])

        print(f"\n--- Accuracy (Python vs R, Pearson r on p.value) ---")
        print(f"  olink_ttest  r = {tt_corr:.4f}")
        print(f"  olink_wilcox r = {ww_corr:.4f}")

        summary.update({
            "r_ttest_ms": r_t * 1000,
            "r_wilcox_ms": r_w * 1000,
            "r_total_wall_ms": r_total * 1000,
            "ttest_pearson_r": tt_corr,
            "wilcox_pearson_r": ww_corr,
        })
    else:
        print("\n[R OlinkAnalyze unavailable — skipping R comparison]")
        summary["r_available"] = False

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report -> {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
