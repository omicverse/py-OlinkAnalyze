"""Shared synthetic NPX-data generator for tests and benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_npx(
    n_proteins: int = 50,
    n_samples_per_group: int = 8,
    n_groups: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic Olink long-format NPX DataFrame with planted DE.

    Two groups (``group0``, ``group1``) with ~5% of proteins
    differentially expressed (effect size ±1.5 NPX, randomly chosen
    sign).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_samples_per_group * n_groups
    rows = []
    base = rng.normal(0.0, 2.0, n_proteins)
    sigma = rng.uniform(0.2, 0.8, n_proteins)
    delta = rng.choice([-1, 0, 1], n_proteins, p=[0.05, 0.9, 0.05]) * 1.5
    for prot_i in range(n_proteins):
        for s_i in range(n_samples):
            g = s_i // n_samples_per_group
            mu = base[prot_i] + (delta[prot_i] if g == 1 else 0.0)
            npx = rng.normal(mu, sigma[prot_i])
            rows.append({
                "SampleID":   f"S{s_i:03d}",
                "OlinkID":    f"OID{prot_i:05d}",
                "UniProt":    f"P{prot_i:05d}",
                "Assay":      f"prot_{prot_i:04d}",
                "Panel":      "Inflammation",
                "NPX":        float(npx),
                "QC_Warning": "PASS",
                "LOD":        float(rng.uniform(-5, -2)),
                "Treatment":  f"group{g}",
            })
    return pd.DataFrame(rows)


def generate_bridge_pair(
    n_proteins: int = 30,
    n_samples_per_project: int = 12,
    n_bridge: int = 4,
    batch_shift_std: float = 0.3,
    seed: int = 0,
):
    """Two NPX frames with a built-in additive per-assay batch effect.

    Returns
    -------
    (df_ref, df_target, bridge_samples, true_shifts)
        ``true_shifts`` is the per-OlinkID shift applied to ``df_target``.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.5, n_proteins)
    sigma = rng.uniform(0.2, 0.5, n_proteins)
    batch_shift = rng.normal(0.0, batch_shift_std, n_proteins)

    def _make(project_name: str, sample_ids, batch=False):
        rows = []
        for s_i, sid in enumerate(sample_ids):
            for prot_i in range(n_proteins):
                mu = base[prot_i] + (batch_shift[prot_i] if batch else 0.0)
                npx = rng.normal(mu, sigma[prot_i])
                rows.append({
                    "SampleID":   sid,
                    "OlinkID":    f"OID{prot_i:05d}",
                    "UniProt":    f"P{prot_i:05d}",
                    "Assay":      f"prot_{prot_i:04d}",
                    "Panel":      "Inflammation",
                    "NPX":        float(npx),
                    "QC_Warning": "PASS",
                    "LOD":        -3.0,
                    "Project":    project_name,
                })
        return pd.DataFrame(rows)

    bridge_ids = [f"B{i:02d}" for i in range(n_bridge)]
    ref_ids = bridge_ids + [f"R{i:02d}" for i in range(n_samples_per_project - n_bridge)]
    tgt_ids = bridge_ids + [f"T{i:02d}" for i in range(n_samples_per_project - n_bridge)]
    df_ref = _make("ref", ref_ids, batch=False)
    df_target = _make("target", tgt_ids, batch=True)
    return df_ref, df_target, bridge_ids, batch_shift
