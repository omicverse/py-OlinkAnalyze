"""Pathway / gene-set enrichment for Olink DE results.

Port of ``OlinkAnalyze::olink_pathway_enrichment``. The R version wraps
``clusterProfiler`` + ``msigdbr``; here we implement a self-contained,
dependency-light (numpy / scipy / pandas only) equivalent:

* **GSEA** — preranked gene-set enrichment with the classic
  Subramanian et al. (2005) weighted Kolmogorov-Smirnov enrichment
  score and a permutation null for the p-value / NES.
* **ORA** — over-representation analysis via the hypergeometric test
  on the set of significant genes.

Gene sets are passed as a ``dict[str, list[str]]`` (term -> gene
symbols). When you have a GMT file use :func:`read_gmt`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as scistats


def read_gmt(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Read a GMT gene-set file into ``{term: [genes]}``."""
    sets: Dict[str, List[str]] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


# ----------------------------------------------------------------------
# Preranked GSEA
# ----------------------------------------------------------------------

def _running_es(ranked_genes: np.ndarray, ranked_scores: np.ndarray,
                gene_set: set, weight: float = 1.0):
    """Weighted Kolmogorov-Smirnov running enrichment score.

    Returns ``(es, running)`` — the maximum-deviation ES and the full
    running-sum array.
    """
    n = len(ranked_genes)
    hits = np.array([g in gene_set for g in ranked_genes])
    n_hit = int(hits.sum())
    if n_hit == 0 or n_hit == n:
        return 0.0, np.zeros(n)
    hit_w = (np.abs(ranked_scores) ** weight) * hits
    norm_hit = hit_w.sum()
    if norm_hit == 0:
        return 0.0, np.zeros(n)
    p_hit = np.cumsum(hit_w) / norm_hit
    p_miss = np.cumsum(~hits) / (n - n_hit)
    running = p_hit - p_miss
    es = running[np.argmax(np.abs(running))]
    return float(es), running


def _gsea_preranked(
    ranking: pd.Series,
    gene_sets: Dict[str, List[str]],
    min_size: int = 5,
    max_size: int = 500,
    n_perm: int = 1000,
    weight: float = 1.0,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Preranked GSEA with a permutation null on the gene labels."""
    rng = np.random.default_rng(seed)
    ranking = ranking.dropna().sort_values(ascending=False)
    genes = ranking.index.to_numpy()
    scores = ranking.to_numpy(dtype=float)
    universe = set(genes)

    rows = []
    for term, members in gene_sets.items():
        gs = universe.intersection(members)
        if not (min_size <= len(gs) <= max_size):
            continue
        es, _ = _running_es(genes, scores, gs, weight)
        # Permutation null: shuffle the gene labels.
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(len(genes))
            null[i], _ = _running_es(genes[perm], scores, gs, weight)
        # Normalized ES against the same-sign tail of the null.
        same_sign = null[np.sign(null) == np.sign(es)] if es != 0 else null
        mean_abs = np.mean(np.abs(same_sign)) if len(same_sign) else np.nan
        nes = es / mean_abs if mean_abs and mean_abs > 0 else np.nan
        if es >= 0:
            p = (np.sum(null >= es) + 1) / (n_perm + 1)
        else:
            p = (np.sum(null <= es) + 1) / (n_perm + 1)
        rows.append({
            "Description": term, "setSize": len(gs),
            "enrichmentScore": es, "NES": nes, "pvalue": p,
            "core_enrichment": ",".join(sorted(gs)),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p.adjust"] = _bh(res["pvalue"].to_numpy())
    return res.sort_values("pvalue").reset_index(drop=True)


# ----------------------------------------------------------------------
# Over-representation analysis
# ----------------------------------------------------------------------

def _ora(
    sig_genes: Sequence[str],
    universe: Sequence[str],
    gene_sets: Dict[str, List[str]],
    min_size: int = 5,
    max_size: int = 500,
) -> pd.DataFrame:
    """Hypergeometric over-representation analysis."""
    sig = set(sig_genes)
    uni = set(universe)
    n_universe = len(uni)
    n_sig = len(sig & uni)
    rows = []
    for term, members in gene_sets.items():
        gs = uni.intersection(members)
        if not (min_size <= len(gs) <= max_size):
            continue
        overlap = sig & gs
        k = len(overlap)
        # P(X >= k) hypergeometric.
        p = scistats.hypergeom.sf(k - 1, n_universe, len(gs), n_sig)
        rows.append({
            "Description": term, "setSize": len(gs),
            "Count": k,
            "GeneRatio": f"{k}/{n_sig}" if n_sig else "0/0",
            "BgRatio": f"{len(gs)}/{n_universe}",
            "pvalue": float(p),
            "geneID": ",".join(sorted(overlap)),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p.adjust"] = _bh(res["pvalue"].to_numpy())
    return res.sort_values("pvalue").reset_index(drop=True)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = p * n / ranks
    s = adj[order]
    s = np.minimum.accumulate(s[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(s, 0.0, 1.0)
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def olink_pathway_enrichment(
    test_results: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    method: str = "gsea",
    gene_col: str = "Assay",
    estimate_col: str = "estimate",
    p_col: str = "p.value",
    pvalue_cutoff: float = 0.05,
    estimate_cutoff: float = 0.0,
    n_perm: int = 1000,
    min_size: int = 5,
    max_size: int = 500,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Gene-set enrichment on an Olink differential-expression result.

    Port of ``OlinkAnalyze::olink_pathway_enrichment``.

    Parameters
    ----------
    test_results :
        Output of :func:`olink_ttest` / :func:`olink_anova` etc. — must
        contain a gene-symbol column and an effect-estimate column.
    gene_sets :
        ``{term: [gene symbols]}`` — e.g. :func:`read_gmt` output, or
        omicverse's predefined signatures.
    method :
        ``'gsea'`` (preranked GSEA, default) or ``'ora'``
        (over-representation analysis on significant genes).
    gene_col :
        Column holding the gene symbol (default ``Assay``).
    estimate_col :
        Effect-size column — used for the GSEA ranking metric (and for
        the ORA significance filter together with ``p_col``).
    pvalue_cutoff, estimate_cutoff :
        ORA significance thresholds (ignored for GSEA).

    Returns
    -------
    pandas.DataFrame
        GSEA: ``Description, setSize, enrichmentScore, NES, pvalue,
        p.adjust, core_enrichment``.
        ORA: ``Description, setSize, Count, GeneRatio, BgRatio, pvalue,
        p.adjust, geneID``.
    """
    method = method.lower()
    if method not in ("gsea", "ora"):
        raise ValueError("method must be 'gsea' or 'ora'.")
    if estimate_col not in test_results.columns:
        raise ValueError(
            f"estimate column '{estimate_col}' missing from test_results."
        )
    if "contrast" in test_results.columns \
            and test_results["contrast"].nunique() > 1:
        raise ValueError(
            "More than one contrast in test_results; filter first."
        )

    tr = test_results.dropna(subset=[gene_col, estimate_col]).copy()
    tr[gene_col] = tr[gene_col].astype(str)
    # Collapse duplicate genes by the largest |estimate|.
    tr["_abs"] = tr[estimate_col].abs()
    tr = tr.sort_values("_abs", ascending=False).drop_duplicates(gene_col)

    if method == "gsea":
        # Signed ranking metric: -log10(p) * sign(estimate) if p exists,
        # else the estimate itself.
        if p_col in tr.columns and tr[p_col].notna().any():
            metric = (-np.log10(tr[p_col].clip(lower=1e-300))
                      * np.sign(tr[estimate_col]))
        else:
            metric = tr[estimate_col]
        ranking = pd.Series(metric.to_numpy(), index=tr[gene_col].to_numpy())
        return _gsea_preranked(ranking, gene_sets, min_size=min_size,
                               max_size=max_size, n_perm=n_perm, seed=seed)
    else:
        universe = tr[gene_col].tolist()
        sig_mask = tr[estimate_col].abs() > estimate_cutoff
        if p_col in tr.columns:
            sig_mask &= tr[p_col] < pvalue_cutoff
        sig = tr.loc[sig_mask, gene_col].tolist()
        return _ora(sig, universe, gene_sets,
                    min_size=min_size, max_size=max_size)
