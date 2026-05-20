"""Per-protein differential expression for Olink NPX data.

Three test families, mirroring the R reference:

* :func:`olink_ttest`  — ``OlinkAnalyze::olink_ttest``. Welch two-sample
  t-test (unequal variances) on NPX per OlinkID, BH-adjusted.
* :func:`olink_wilcox` — ``OlinkAnalyze::olink_wilcox``. Mann-Whitney /
  Wilcoxon rank-sum test (``exact=FALSE, correct=TRUE`` in R, which is
  scipy's ``mannwhitneyu(use_continuity=True)``).
* :func:`olink_lmer`   — ``OlinkAnalyze::olink_lmer``. Linear
  mixed-effects model per OlinkID with one or more random effects,
  fitted via ``statsmodels.regression.mixed_linear_model.MixedLM``.

All three return a long-form DataFrame with one row per OlinkID and
the same column schema:

    OlinkID, Assay, UniProt, term, estimate, statistic,
    p.value, Adjusted_pval, Threshold

``term`` is the contrast / covariate label (e.g. ``"group1 - group0"``
for t-test, the random-effect-free fixed-effect coefficient for LMM).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as scistats


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment (matches ``p.adjust(method='BH')``)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    # Mask NaNs but keep them in the output.
    mask = ~np.isnan(p)
    p_valid = p[mask]
    m = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adj = p_valid * m / ranks
    # Enforce monotonicity over the sorted (ascending) p-values.
    sorted_adj = adj[order]
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    out_valid = np.empty(m)
    out_valid[order] = np.clip(sorted_adj, 0.0, 1.0)
    out = np.full(n, np.nan)
    out[mask] = out_valid
    return out


def _assay_meta(group: pd.DataFrame) -> dict:
    """Pull the first available Assay / UniProt / Panel metadata."""
    return {
        "Assay": group["Assay"].iloc[0] if "Assay" in group.columns else "",
        "UniProt": group["UniProt"].iloc[0] if "UniProt" in group.columns else "",
        "Panel": group["Panel"].iloc[0] if "Panel" in group.columns else "",
    }


# ----------------------------------------------------------------------
# t-test
# ----------------------------------------------------------------------

def olink_ttest(
    df: pd.DataFrame,
    variable: str,
    pair_id: Optional[str] = None,
    alternative: str = "two-sided",
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein Welch two-sample t-test on NPX (paired if ``pair_id``).

    Parameters
    ----------
    df :
        Long-format NPX frame.
    variable :
        Column containing the two-level grouping factor.
    pair_id :
        If given, run a paired t-test (each level pairs samples by this
        column).
    alternative :
        ``"two-sided"`` (default), ``"greater"`` or ``"less"``.
    npx_col, assay_col :
        Column overrides.
    threshold :
        Significance threshold for the ``Threshold`` flag column.
    """
    levels = sorted(df[variable].dropna().unique())
    if len(levels) != 2:
        raise ValueError(
            f"olink_ttest requires exactly 2 levels in '{variable}'; got {levels}"
        )
    g0, g1 = levels
    term_label = f"{g1} - {g0}"

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        x0 = group.loc[group[variable] == g0, npx_col].to_numpy(dtype=float)
        x1 = group.loc[group[variable] == g1, npx_col].to_numpy(dtype=float)
        x0 = x0[~np.isnan(x0)]
        x1 = x1[~np.isnan(x1)]

        meta = _assay_meta(group)
        if pair_id is not None:
            # Paired: match by pair_id
            sub = group.dropna(subset=[npx_col, pair_id, variable])
            pivot = sub.pivot_table(index=pair_id, columns=variable,
                                    values=npx_col, aggfunc="mean")
            pivot = pivot.dropna(subset=[g0, g1])
            if len(pivot) < 2:
                stat, p, est = np.nan, np.nan, np.nan
            else:
                stat, p = scistats.ttest_rel(
                    pivot[g1].to_numpy(), pivot[g0].to_numpy(),
                    alternative=alternative,
                )
                est = float(np.mean(pivot[g1].to_numpy() - pivot[g0].to_numpy()))
        else:
            if len(x0) < 2 or len(x1) < 2:
                stat, p, est = np.nan, np.nan, np.nan
            else:
                stat, p = scistats.ttest_ind(
                    x1, x0, equal_var=False, alternative=alternative,
                )
                est = float(np.mean(x1) - np.mean(x0))

        rows.append({
            assay_col: oid,
            **meta,
            "term": term_label,
            "estimate": est,
            "statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p.value": float(p) if not np.isnan(p) else np.nan,
        })

    res = pd.DataFrame(rows)
    res["Adjusted_pval"] = _bh_adjust(res["p.value"].to_numpy())
    res["Threshold"] = (res["Adjusted_pval"] < threshold).astype(int)
    return res.sort_values("p.value", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# Wilcoxon
# ----------------------------------------------------------------------

def olink_wilcox(
    df: pd.DataFrame,
    variable: str,
    alternative: str = "two-sided",
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein Mann-Whitney / Wilcoxon rank-sum test on NPX.

    Matches R's ``wilcox.test(exact = FALSE, correct = TRUE)`` —
    i.e. asymptotic Mann-Whitney with continuity correction.
    """
    levels = sorted(df[variable].dropna().unique())
    if len(levels) != 2:
        raise ValueError(
            f"olink_wilcox requires exactly 2 levels in '{variable}'; got {levels}"
        )
    g0, g1 = levels
    term_label = f"{g1} - {g0}"

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        x0 = group.loc[group[variable] == g0, npx_col].to_numpy(dtype=float)
        x1 = group.loc[group[variable] == g1, npx_col].to_numpy(dtype=float)
        x0 = x0[~np.isnan(x0)]
        x1 = x1[~np.isnan(x1)]
        meta = _assay_meta(group)
        if len(x0) < 2 or len(x1) < 2:
            stat, p, est = np.nan, np.nan, np.nan
        else:
            stat, p = scistats.mannwhitneyu(
                x1, x0,
                use_continuity=True,
                alternative=alternative,
                method="asymptotic",
            )
            est = float(np.median(x1) - np.median(x0))
        rows.append({
            assay_col: oid,
            **meta,
            "term": term_label,
            "estimate": est,
            "statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p.value": float(p) if not np.isnan(p) else np.nan,
        })
    res = pd.DataFrame(rows)
    res["Adjusted_pval"] = _bh_adjust(res["p.value"].to_numpy())
    res["Threshold"] = (res["Adjusted_pval"] < threshold).astype(int)
    return res.sort_values("p.value", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# Linear mixed-effects
# ----------------------------------------------------------------------

def olink_lmer(
    df: pd.DataFrame,
    variable: Union[str, Sequence[str]],
    random: Union[str, Sequence[str]],
    covariates: Optional[Sequence[str]] = None,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein linear mixed-effects model.

    Models ``NPX ~ variable + covariates + (1 | random)`` per OlinkID
    using :class:`statsmodels.regression.mixed_linear_model.MixedLM`.

    Parameters
    ----------
    variable :
        Fixed-effect term(s) of interest (string or list).
    random :
        Random-effect grouping variable (single column for ``(1|g)``-style).
        If a list is given, the *first* element is used as the grouping
        variable and the rest as nested fixed effects (matches the
        common ``olink_lmer`` invocation; full multi-grouping is v0.2).
    covariates :
        Optional additional fixed-effect columns.
    """
    import statsmodels.formula.api as smf

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(random, str):
        random_group = random
    else:
        random_group = list(random)[0]

    fixed_terms = list(variable) + list(covariates or [])
    formula = f"{npx_col} ~ " + " + ".join(fixed_terms)

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=[npx_col, random_group] + fixed_terms).copy()
        meta = _assay_meta(group)
        if sub[random_group].nunique() < 2 or len(sub) < 3:
            for term in fixed_terms:
                rows.append({assay_col: oid, **meta, "term": term,
                             "estimate": np.nan, "statistic": np.nan,
                             "p.value": np.nan})
            continue
        try:
            mod = smf.mixedlm(formula, sub, groups=sub[random_group])
            fit = mod.fit(reml=False, method="lbfgs",
                          disp=False, full_output=False)
            params = fit.params
            tvals = fit.tvalues
            pvals = fit.pvalues
            for term_name in params.index:
                if term_name in ("Intercept", "Group Var"):
                    continue
                # Map back to a "clean" term label by stripping the
                # category contrast suffix (e.g. ``variable[T.b]`` -> ``variable``).
                clean = term_name.split("[T.")[0]
                rows.append({
                    assay_col: oid, **meta,
                    "term": clean,
                    "estimate": float(params[term_name]),
                    "statistic": float(tvals[term_name]),
                    "p.value": float(pvals[term_name]),
                })
        except Exception:
            for term in fixed_terms:
                rows.append({assay_col: oid, **meta, "term": term,
                             "estimate": np.nan, "statistic": np.nan,
                             "p.value": np.nan})

    res = pd.DataFrame(rows)
    # BH-adjust per term across proteins.
    res["Adjusted_pval"] = np.nan
    for term, idx in res.groupby("term").groups.items():
        res.loc[idx, "Adjusted_pval"] = _bh_adjust(
            res.loc[idx, "p.value"].to_numpy()
        )
    res["Threshold"] = (res["Adjusted_pval"] < threshold).astype(int)
    return res.sort_values(["term", "p.value"], na_position="last").reset_index(drop=True)
