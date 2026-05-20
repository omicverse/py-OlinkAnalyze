"""Per-protein differential expression for Olink NPX data.

Test families mirroring the R reference:

* :func:`olink_ttest`  — ``OlinkAnalyze::olink_ttest``. Welch two-sample
  t-test (unequal variances) on NPX per OlinkID, BH-adjusted.
* :func:`olink_wilcox` — ``OlinkAnalyze::olink_wilcox``. Mann-Whitney /
  Wilcoxon rank-sum test (``exact=FALSE, correct=TRUE`` in R, which is
  scipy's ``mannwhitneyu(use_continuity=True)``).
* :func:`olink_lmer`   — ``OlinkAnalyze::olink_lmer``. Linear
  mixed-effects model per OlinkID with one or more random effects,
  fitted via ``statsmodels.regression.mixed_linear_model.MixedLM``.
* :func:`olink_anova`  — ``OlinkAnalyze::olink_anova``. One-/multi-way
  type-III ANOVA per OlinkID (``car::Anova`` parity via ``contr.sum``).
* :func:`olink_anova_posthoc` — Tukey HSD post-hoc contrasts.
* :func:`olink_one_non_parametric` — Kruskal-Wallis / Friedman test.
* :func:`olink_one_non_parametric_posthoc` — Dunn / paired-Wilcoxon
  post-hoc.
* :func:`olink_ordinal_regression` — proportional-odds ordinal
  regression per OlinkID.
* :func:`olink_ordinal_regression_posthoc` — ordinal-model post-hoc.
* :func:`olink_lmer_posthoc` — LMM post-hoc contrasts.

Two-sample tests return a long-form DataFrame with one row per OlinkID:

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


# ----------------------------------------------------------------------
# ANOVA (one-way / multi-way, type-III)
# ----------------------------------------------------------------------

def _bh_adjust_grouped(res: pd.DataFrame, key: str, p_col: str = "p.value",
                       out_col: str = "Adjusted_pval") -> pd.DataFrame:
    """BH-adjust ``p_col`` independently within each value of ``key``."""
    res[out_col] = np.nan
    for _, idx in res.groupby(key).groups.items():
        res.loc[idx, out_col] = _bh_adjust(res.loc[idx, p_col].to_numpy())
    return res


def olink_anova(
    df: pd.DataFrame,
    variable: Union[str, Sequence[str]],
    covariates: Optional[Sequence[str]] = None,
    outcome: str = "NPX",
    return_covariates: bool = False,
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein type-III ANOVA — port of ``OlinkAnalyze::olink_anova``.

    Fits ``outcome ~ variable [* variable2 ...] [+ covariates]`` per
    OlinkID with ``statsmodels.formula.api.ols`` and computes the
    type-III sum-of-squares ANOVA table via
    ``statsmodels.stats.anova.anova_lm(typ=3)``. To match R's
    ``car::Anova(type=3)`` the categorical factors are coded with
    sum-to-zero contrasts (``contr.sum`` <-> ``Sum`` in patsy).

    Parameters
    ----------
    variable :
        Factor(s) of interest. A list yields a crossed (``*``) model.
    covariates :
        Optional adjustment terms (added with ``+``); their p-values
        are reported but not BH-adjusted (matches R).
    return_covariates :
        If ``True`` keep covariate rows in the output.

    Returns
    -------
    pandas.DataFrame with columns
        ``OlinkID, Assay, UniProt, Panel, term, df, sumsq, meansq,
        statistic, p.value, Adjusted_pval, Threshold``.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    if isinstance(variable, str):
        variable = [variable]
    variable = list(variable)
    covariates = list(covariates or [])

    # Build the formula. Crossed variables, additive covariates.
    var_part = " * ".join(f"C({v}, Sum)" for v in variable)
    cov_part = " + ".join(f"C({c}, Sum)" for c in covariates)
    rhs = var_part + (" + " + cov_part if cov_part else "")
    formula = f"{outcome} ~ {rhs}"

    # term-name -> clean label map (strip the C(.., Sum) wrapper).
    def _clean(term: str) -> str:
        return term.replace("C(", "").replace(", Sum)", "").replace(":", ":")

    cov_clean = set(covariates)
    rows = []
    needed = [outcome] + variable + covariates
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=needed).copy()
        meta = _assay_meta(group)
        try:
            fit = smf.ols(formula, data=sub).fit()
            tbl = anova_lm(fit, typ=3)
        except Exception:
            continue
        for term, r in tbl.iterrows():
            if term in ("Intercept", "Residual", "Residuals"):
                continue
            clean = _clean(str(term))
            rows.append({
                assay_col: oid, **meta, "term": clean,
                "df": float(r.get("df", np.nan)),
                "sumsq": float(r.get("sum_sq", np.nan)),
                "statistic": float(r.get("F", np.nan)),
                "p.value": float(r.get("PR(>F)", np.nan)),
                "_is_cov": clean in cov_clean,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["meansq"] = res["sumsq"] / res["df"]
    # BH-adjust separately for covariate / non-covariate groups, then
    # blank out covariate adjusted p-values (R behaviour).
    res = _bh_adjust_grouped(res, "_is_cov")
    res.loc[res["_is_cov"], "Adjusted_pval"] = np.nan
    res["Threshold"] = np.where(
        res["_is_cov"], None,
        np.where(res["Adjusted_pval"] < threshold,
                 "Significant", "Non-significant"),
    )
    if not return_covariates:
        res = res.loc[~res["_is_cov"]]
    res = res.drop(columns=["_is_cov"])
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "df",
            "sumsq", "meansq", "statistic", "p.value",
            "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)


def olink_anova_posthoc(
    df: pd.DataFrame,
    variable: Union[str, Sequence[str]],
    effect: str,
    outcome: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Tukey-HSD post-hoc for a one-way ANOVA factor.

    Port of ``OlinkAnalyze::olink_anova_posthoc`` (pairwise contrasts).
    Uses ``statsmodels.stats.multicomp.pairwise_tukeyhsd`` per OlinkID.
    The ``effect`` must be one of the ``variable`` factors.

    Returns one row per (OlinkID, pairwise contrast) with columns
    ``OlinkID, Assay, UniProt, Panel, term, contrast, estimate,
    conf.low, conf.high, Adjusted_pval, Threshold``.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    if isinstance(variable, str):
        variable = [variable]
    if effect not in variable:
        raise ValueError(f"effect '{effect}' must be one of variable {variable}")

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=[outcome, effect]).copy()
        meta = _assay_meta(group)
        if sub[effect].nunique() < 2:
            continue
        try:
            tk = pairwise_tukeyhsd(sub[outcome].to_numpy(),
                                   sub[effect].astype(str).to_numpy())
        except Exception:
            continue
        # tk._results_table.data[0] is the header.
        for r in tk._results_table.data[1:]:
            g1, g2, meandiff, padj, lo, hi, _rej = r
            rows.append({
                assay_col: oid, **meta, "term": effect,
                "contrast": f"{g1} - {g2}",
                "estimate": float(meandiff),
                "conf.low": float(lo),
                "conf.high": float(hi),
                "Adjusted_pval": float(padj),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["Threshold"] = np.where(res["Adjusted_pval"] < threshold,
                                "Significant", "Non-significant")
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# Non-parametric one-way (Kruskal-Wallis / Friedman)
# ----------------------------------------------------------------------

def olink_one_non_parametric(
    df: pd.DataFrame,
    variable: str,
    dependence: bool = False,
    subject: Optional[str] = None,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein Kruskal-Wallis (independent) / Friedman (paired) test.

    Port of ``OlinkAnalyze::olink_one_non_parametric``.

    * ``dependence=False`` -> ``scipy.stats.kruskal`` (>= 2 groups).
    * ``dependence=True``  -> ``scipy.stats.friedmanchisquare`` — needs
      a ``subject`` column; subjects with incomplete data are dropped.

    Returns ``OlinkID, Assay, UniProt, Panel, term, df, method,
    statistic, p.value, Adjusted_pval, Threshold``.
    """
    levels = sorted(df[variable].dropna().unique())
    if len(levels) < 2:
        raise ValueError(
            f"olink_one_non_parametric requires >= 2 levels in '{variable}'."
        )
    if dependence and subject is None:
        raise ValueError("subject must be specified when dependence=True.")

    method = "Friedman rank sum test" if dependence else "Kruskal-Wallis rank sum test"
    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        meta = _assay_meta(group)
        sub = group.dropna(subset=[npx_col, variable])
        if dependence:
            sub = sub.dropna(subset=[subject])
            pivot = sub.pivot_table(index=subject, columns=variable,
                                    values=npx_col, aggfunc="mean")
            pivot = pivot.dropna()
            if pivot.shape[0] < 2 or pivot.shape[1] < 2:
                stat, p = np.nan, np.nan
            else:
                stat, p = scistats.friedmanchisquare(
                    *[pivot[c].to_numpy() for c in pivot.columns]
                )
            ddf = float(len(levels) - 1)
        else:
            samples = [sub.loc[sub[variable] == lv, npx_col].to_numpy()
                       for lv in levels]
            samples = [s[~np.isnan(s)] for s in samples]
            if any(len(s) < 1 for s in samples) or sum(len(s) for s in samples) < 3:
                stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = scistats.kruskal(*samples)
                except ValueError:
                    stat, p = np.nan, np.nan
            ddf = float(len(levels) - 1)
        rows.append({
            assay_col: oid, **meta, "term": variable, "df": ddf,
            "method": method,
            "statistic": float(stat) if stat == stat else np.nan,
            "p.value": float(p) if p == p else np.nan,
        })
    res = pd.DataFrame(rows)
    res["Adjusted_pval"] = _bh_adjust(res["p.value"].to_numpy())
    res["Threshold"] = np.where(res["Adjusted_pval"] < threshold,
                                "Significant", "Non-significant")
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "df",
            "method", "statistic", "p.value", "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)


def _dunn_test(values: np.ndarray, groups: np.ndarray, labels: Sequence):
    """Dunn's test of multiple comparisons after Kruskal-Wallis.

    Returns a list of ``(g1, g2, Z, p_raw)`` tuples — p-values are raw
    (BH correction applied by the caller, matching R's ``method='bh'``).
    """
    n = len(values)
    ranks = scistats.rankdata(values)
    # Tie correction term.
    _, counts = np.unique(values, return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    sigma2 = (n * (n + 1) / 12.0) - ties / (12.0 * (n - 1))
    out = []
    grp_n = {g: int(np.sum(groups == g)) for g in labels}
    grp_rsum = {g: float(np.sum(ranks[groups == g])) for g in labels}
    grp_rmean = {g: grp_rsum[g] / grp_n[g] for g in labels if grp_n[g] > 0}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            if grp_n.get(a, 0) == 0 or grp_n.get(b, 0) == 0:
                out.append((a, b, np.nan, np.nan))
                continue
            se = np.sqrt(sigma2 * (1.0 / grp_n[a] + 1.0 / grp_n[b]))
            z = (grp_rmean[a] - grp_rmean[b]) / se if se > 0 else np.nan
            p = 2.0 * scistats.norm.sf(abs(z)) if z == z else np.nan
            out.append((a, b, float(z) if z == z else np.nan,
                        float(p) if p == p else np.nan))
    return out


def olink_one_non_parametric_posthoc(
    df: pd.DataFrame,
    variable: str,
    test: str = "kruskal",
    subject: Optional[str] = None,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Post-hoc pairwise tests after a non-parametric one-way analysis.

    Port of ``OlinkAnalyze::olink_one_non_parametric_posthoc``.

    * ``test='kruskal'``  -> Dunn's test, BH-corrected (``estimate`` is
      the Z statistic).
    * ``test='friedman'`` -> paired Wilcoxon signed-rank tests,
      BH-corrected (needs ``subject``).
    """
    if test not in ("kruskal", "friedman"):
        raise ValueError("test must be 'kruskal' or 'friedman'.")
    if test == "friedman" and subject is None:
        raise ValueError("subject must be specified for test='friedman'.")
    levels = sorted(df[variable].dropna().unique())

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        meta = _assay_meta(group)
        sub = group.dropna(subset=[npx_col, variable])
        if test == "kruskal":
            sub = sub.dropna(subset=[npx_col])
            vals = sub[npx_col].to_numpy(dtype=float)
            grps = sub[variable].astype(str).to_numpy()
            present = [str(lv) for lv in levels if str(lv) in set(grps)]
            if len(present) < 2 or len(vals) < 3:
                continue
            for g1, g2, z, p in _dunn_test(vals, grps, present):
                rows.append({
                    assay_col: oid, **meta, "term": variable,
                    "contrast": f"{g1} - {g2}",
                    "estimate": z, "p.value": p,
                })
        else:
            sub = sub.dropna(subset=[subject])
            pivot = sub.pivot_table(index=subject, columns=variable,
                                    values=npx_col, aggfunc="mean").dropna()
            present = [lv for lv in levels if lv in pivot.columns]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
                    if len(x) < 2:
                        stat, p, est = np.nan, np.nan, np.nan
                    else:
                        try:
                            stat, p = scistats.wilcoxon(x, y)
                        except ValueError:
                            stat, p = np.nan, np.nan
                        est = float(np.median(x - y))
                    rows.append({
                        assay_col: oid, **meta, "term": variable,
                        "contrast": f"{a} - {b}",
                        "estimate": est, "p.value": p,
                    })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # BH-correct within each (OlinkID) group.
    res = _bh_adjust_grouped(res, assay_col)
    res["Threshold"] = np.where(res["Adjusted_pval"] < threshold,
                                "Significant", "Non-significant")
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "contrast",
            "estimate", "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# Ordinal regression (proportional odds)
# ----------------------------------------------------------------------

def olink_ordinal_regression(
    df: pd.DataFrame,
    variable: Union[str, Sequence[str]],
    covariates: Optional[Sequence[str]] = None,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    return_covariates: bool = False,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-protein proportional-odds ordinal regression.

    Port of ``OlinkAnalyze::olink_ordinalRegression``. R ranks the NPX
    values per assay and treats the integer ranks as ordered response
    levels, then fits ``ordinal::clm`` and runs a type-III ANOVA
    (likelihood-ratio chi-square per term). We mirror this with
    :class:`statsmodels.miscmodels.ordinal_model.OrderedModel` (logit
    link) and compute a likelihood-ratio test per term.

    Returns ``OlinkID, Assay, UniProt, Panel, term, df, statistic,
    p.value, Adjusted_pval, Threshold``.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    if isinstance(variable, str):
        variable = [variable]
    variable = list(variable)
    covariates = list(covariates or [])
    terms = variable + covariates
    cov_clean = set(covariates)

    def _design(sub, drop_term=None):
        cols = []
        for t in terms:
            if t == drop_term:
                continue
            s = sub[t]
            if s.dtype == object or str(s.dtype).startswith("category"):
                dummies = pd.get_dummies(s.astype(str), prefix=t,
                                         drop_first=True, dtype=float)
                cols.append(dummies)
            else:
                cols.append(pd.to_numeric(s, errors="coerce").rename(t))
        if not cols:
            return pd.DataFrame(index=sub.index)
        return pd.concat(cols, axis=1)

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=[npx_col] + terms).copy()
        meta = _assay_meta(group)
        if len(sub) < len(terms) + 3:
            continue
        # Ranked NPX -> ordered categorical response.
        y = scistats.rankdata(sub[npx_col].to_numpy(), method="dense")
        y = pd.Categorical(y, ordered=True)
        try:
            X_full = _design(sub)
            full = OrderedModel(y, X_full, distr="logit")
            full_fit = full.fit(method="bfgs", disp=False, maxiter=200)
        except Exception:
            continue
        for t in terms:
            try:
                X_red = _design(sub, drop_term=t)
                if X_red.shape[1] == 0:
                    # Null model: only thresholds.
                    red = OrderedModel(y, np.zeros((len(y), 0)), distr="logit")
                else:
                    red = OrderedModel(y, X_red, distr="logit")
                red_fit = red.fit(method="bfgs", disp=False, maxiter=200)
                lr = 2.0 * (full_fit.llf - red_fit.llf)
                ddf = X_full.shape[1] - X_red.shape[1]
                p = scistats.chi2.sf(lr, ddf) if ddf > 0 else np.nan
            except Exception:
                lr, ddf, p = np.nan, np.nan, np.nan
            rows.append({
                assay_col: oid, **meta, "term": t,
                "df": float(ddf) if ddf == ddf else np.nan,
                "statistic": float(lr) if lr == lr else np.nan,
                "p.value": float(p) if p == p else np.nan,
                "_is_cov": t in cov_clean,
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = _bh_adjust_grouped(res, "_is_cov")
    res.loc[res["_is_cov"], "Adjusted_pval"] = np.nan
    res["Threshold"] = np.where(
        res["_is_cov"], None,
        np.where(res["Adjusted_pval"] < threshold,
                 "Significant", "Non-significant"),
    )
    if not return_covariates:
        res = res.loc[~res["_is_cov"]]
    res = res.drop(columns=["_is_cov"])
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "df",
            "statistic", "p.value", "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("p.value", na_position="last").reset_index(drop=True)


def olink_ordinal_regression_posthoc(
    df: pd.DataFrame,
    variable: str,
    effect: str,
    covariates: Optional[Sequence[str]] = None,
    npx_col: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Pairwise post-hoc for the ordinal model.

    Port of ``OlinkAnalyze::olink_ordinalRegression_posthoc``. emmeans
    on a ``clm`` model is not reproducible dependency-light; instead we
    fit per-protein ordinal models and report Wald pairwise contrasts
    of the ``effect`` factor's coefficients (BH-corrected within
    OlinkID). ``estimate`` is the log-odds difference.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    covariates = list(covariates or [])
    levels = sorted(df[variable].dropna().unique()) if effect == variable \
        else sorted(df[effect].dropna().unique())
    terms = [variable] + [c for c in covariates if c != variable]

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=[npx_col] + terms).copy()
        meta = _assay_meta(group)
        present = [lv for lv in levels if lv in set(sub[effect])]
        if len(present) < 2 or len(sub) < len(present) + 3:
            continue
        y = scistats.rankdata(sub[npx_col].to_numpy(), method="dense")
        y = pd.Categorical(y, ordered=True)
        # Effect dummies (treatment coding, reference = first level).
        eff = pd.Categorical(sub[effect].astype(str),
                             categories=[str(p) for p in present], ordered=False)
        dummies = pd.get_dummies(eff, prefix=effect, drop_first=True, dtype=float)
        extra = []
        for c in terms:
            if c == effect:
                continue
            s = sub[c]
            if s.dtype == object:
                extra.append(pd.get_dummies(s.astype(str), prefix=c,
                                            drop_first=True, dtype=float))
            else:
                extra.append(pd.to_numeric(s, errors="coerce").rename(c))
        X = pd.concat([dummies] + extra, axis=1) if extra else dummies
        try:
            fit = OrderedModel(y, X, distr="logit").fit(
                method="bfgs", disp=False, maxiter=200)
        except Exception:
            continue
        params = fit.params
        cov = fit.cov_params()
        # Coefficient for level i vs reference is params[effect_<lvl>].
        ncoef = {str(present[0]): 0.0}
        for k in range(1, len(present)):
            name = f"{effect}_{present[k]}"
            ncoef[str(present[k])] = float(params.get(name, np.nan))
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                na = f"{effect}_{a}" if i > 0 else None
                nb = f"{effect}_{b}" if j > 0 else None
                est = ncoef[str(a)] - ncoef[str(b)]
                # Variance of the contrast.
                var = 0.0
                if na is not None and na in cov.index:
                    var += cov.loc[na, na]
                if nb is not None and nb in cov.index:
                    var += cov.loc[nb, nb]
                if na is not None and nb is not None \
                        and na in cov.index and nb in cov.columns:
                    var -= 2.0 * cov.loc[na, nb]
                se = np.sqrt(var) if var > 0 else np.nan
                z = est / se if se == se and se > 0 else np.nan
                p = 2.0 * scistats.norm.sf(abs(z)) if z == z else np.nan
                rows.append({
                    assay_col: oid, **meta, "term": effect,
                    "contrast": f"{a} - {b}",
                    "estimate": est, "p.value": p,
                })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = _bh_adjust_grouped(res, assay_col)
    res["Threshold"] = np.where(res["Adjusted_pval"] < threshold,
                                "Significant", "Non-significant")
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "contrast",
            "estimate", "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# LMM post-hoc
# ----------------------------------------------------------------------

def olink_lmer_posthoc(
    df: pd.DataFrame,
    variable: Union[str, Sequence[str]],
    random: Union[str, Sequence[str]],
    effect: str,
    covariates: Optional[Sequence[str]] = None,
    outcome: str = "NPX",
    assay_col: str = "OlinkID",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Pairwise post-hoc contrasts for the linear mixed model.

    Port of ``OlinkAnalyze::olink_lmer_posthoc``. R uses
    ``emmeans`` on the fitted ``lmer`` model; here we fit
    ``statsmodels.MixedLM`` per OlinkID and form Wald pairwise
    contrasts of the ``effect`` factor's fixed-effect coefficients
    (BH-corrected within OlinkID). ``estimate`` is the NPX difference.
    """
    import statsmodels.formula.api as smf

    if isinstance(variable, str):
        variable = [variable]
    variable = list(variable)
    random_group = random if isinstance(random, str) else list(random)[0]
    covariates = list(covariates or [])
    fixed = variable + covariates
    levels = sorted(df[effect].dropna().unique())
    formula = f"{outcome} ~ " + " + ".join(fixed)

    rows = []
    for oid, group in df.groupby(assay_col, sort=False):
        sub = group.dropna(subset=[outcome, random_group] + fixed).copy()
        meta = _assay_meta(group)
        present = [lv for lv in levels if lv in set(sub[effect])]
        if sub[random_group].nunique() < 2 or len(present) < 2:
            continue
        try:
            fit = smf.mixedlm(formula, sub, groups=sub[random_group]).fit(
                reml=False, method="lbfgs", disp=False)
        except Exception:
            continue
        params = fit.params
        cov = fit.cov_params()
        # Coefficient for level k vs reference (first level).
        ncoef = {str(present[0]): (0.0, None)}
        for k in range(1, len(present)):
            name = f"{effect}[T.{present[k]}]"
            ncoef[str(present[k])] = (float(params.get(name, np.nan)), name)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                ca, na = ncoef[str(a)]
                cb, nb = ncoef[str(b)]
                est = ca - cb
                var = 0.0
                if na is not None and na in cov.index:
                    var += cov.loc[na, na]
                if nb is not None and nb in cov.index:
                    var += cov.loc[nb, nb]
                if na is not None and nb is not None \
                        and na in cov.index and nb in cov.columns:
                    var -= 2.0 * cov.loc[na, nb]
                se = np.sqrt(var) if var > 0 else np.nan
                t = est / se if se == se and se > 0 else np.nan
                p = 2.0 * scistats.norm.sf(abs(t)) if t == t else np.nan
                rows.append({
                    assay_col: oid, **meta, "term": effect,
                    "contrast": f"{a} - {b}",
                    "estimate": est, "statistic": t, "p.value": p,
                })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = _bh_adjust_grouped(res, assay_col)
    res["Threshold"] = np.where(res["Adjusted_pval"] < threshold,
                                "Significant", "Non-significant")
    cols = [assay_col, "Assay", "UniProt", "Panel", "term", "contrast",
            "estimate", "statistic", "Adjusted_pval", "Threshold"]
    res = res[[c for c in cols if c in res.columns]]
    return res.sort_values("Adjusted_pval", na_position="last").reset_index(drop=True)
