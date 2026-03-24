"""Statistical tests and multiple comparisons correction."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def _clean_paired_row(row_a: np.ndarray, row_b: np.ndarray):
    """Remove pairs where either value is NaN."""
    mask = ~(np.isnan(row_a) | np.isnan(row_b))
    return row_a[mask], row_b[mask]


def _clean_independent_row(row: np.ndarray):
    """Remove NaN values from a single row."""
    return row[~np.isnan(row)]


def permutation_test_paired(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided paired permutation test on mean difference."""
    a, b = _clean_paired_row(a, b)
    if len(a) < 2:
        return np.nan
    diff = a - b
    observed = np.abs(np.mean(diff))
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        if np.abs(np.mean(diff * signs)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_independent(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided independent permutation test on mean difference."""
    a = _clean_independent_row(a)
    b = _clean_independent_row(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = np.abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        if np.abs(np.mean(perm[:na]) - np.mean(perm[na:])) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def run_test_at_timepoint(
    row_a: np.ndarray, row_b: np.ndarray, test_name: str, design: str
) -> tuple[float, float]:
    """Run a statistical test at a single timepoint.

    Returns (test_statistic, p_value).
    """
    if design == "repeated":
        a, b = _clean_paired_row(row_a, row_b)
        if len(a) < 2:
            return np.nan, np.nan

        if test_name == "Permutation test":
            p = permutation_test_paired(row_a, row_b)
            return np.mean(a - b), p
        elif test_name == "Paired t-test":
            stat, p = stats.ttest_rel(a, b)
            return stat, p
        elif test_name == "Wilcoxon signed-rank":
            if len(a) < 6:
                return np.nan, np.nan
            stat, p = stats.wilcoxon(a, b)
            return stat, p
    else:  # independent
        a = _clean_independent_row(row_a)
        b = _clean_independent_row(row_b)
        if len(a) < 2 or len(b) < 2:
            return np.nan, np.nan

        if test_name == "Permutation test":
            p = permutation_test_independent(row_a, row_b)
            return np.mean(a) - np.mean(b), p
        elif test_name == "Independent t-test":
            stat, p = stats.ttest_ind(a, b)
            return stat, p
        elif test_name == "Mann-Whitney U":
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            return stat, p

    return np.nan, np.nan


def run_anova_at_timepoint(
    rows: list[np.ndarray], design: str
) -> tuple[float, float]:
    """Run ANOVA/Kruskal-Wallis/Friedman at a single timepoint for >2 groups.

    Returns (test_statistic, p_value).
    """
    if design == "repeated":
        # Friedman test - needs matched data
        cleaned = []
        # Find common non-NaN across all groups
        mask = np.ones(len(rows[0]), dtype=bool)
        for r in rows:
            mask &= ~np.isnan(r)
        for r in rows:
            cleaned.append(r[mask])
        if len(cleaned[0]) < 3:
            return np.nan, np.nan
        try:
            stat, p = stats.friedmanchisquare(*cleaned)
            return stat, p
        except Exception:
            return np.nan, np.nan
    else:
        # Kruskal-Wallis for independent
        cleaned = [_clean_independent_row(r) for r in rows]
        if any(len(c) < 2 for c in cleaned):
            return np.nan, np.nan
        try:
            stat, p = stats.kruskal(*cleaned)
            return stat, p
        except Exception:
            return np.nan, np.nan


def run_tests_across_timepoints(
    data_a: np.ndarray,
    data_b: np.ndarray,
    test_name: str,
    design: str,
    correction: str = "FDR (Benjamini-Hochberg)",
    alpha: float = 0.05,
) -> dict:
    """Run statistical tests across all timepoints.

    Args:
        data_a: shape (n_timepoints, n_subjects_a)
        data_b: shape (n_timepoints, n_subjects_b)
        test_name: name of statistical test
        design: 'repeated' or 'independent'
        correction: multiple comparisons correction method
        alpha: significance threshold

    Returns dict with arrays: test_stats, p_values, p_corrected, significant
    """
    n_timepoints = data_a.shape[0]
    test_stats = np.full(n_timepoints, np.nan)
    p_values = np.full(n_timepoints, np.nan)

    for t in range(n_timepoints):
        stat, p = run_test_at_timepoint(data_a[t], data_b[t], test_name, design)
        test_stats[t] = stat
        p_values[t] = p

    # Multiple comparisons correction
    p_corrected = apply_correction(p_values, correction, alpha)

    significant = p_corrected < alpha

    return {
        "test_stats": test_stats,
        "p_values": p_values,
        "p_corrected": p_corrected,
        "significant": significant,
    }


def apply_correction(p_values: np.ndarray, method: str, alpha: float = 0.05) -> np.ndarray:
    """Apply multiple comparisons correction to p-values."""
    valid_mask = ~np.isnan(p_values)
    if not valid_mask.any():
        return p_values.copy()

    p_valid = p_values[valid_mask]

    if method == "No correction":
        p_corrected = p_values.copy()
    else:
        method_map = {
            "FDR (Benjamini-Hochberg)": "fdr_bh",
            "Bonferroni": "bonferroni",
            "Holm-Bonferroni": "holm",
        }
        sm_method = method_map.get(method, "fdr_bh")
        _, p_adj, _, _ = multipletests(p_valid, alpha=alpha, method=sm_method)
        p_corrected = np.full_like(p_values, np.nan)
        p_corrected[valid_mask] = p_adj

    return p_corrected


# ── Omnibus tests (ANOVA / LMM / non-parametric) ────────────────────────────

def _complete_cases(long_df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Keep only subjects that have data in every condition."""
    counts = long_df.groupby("Subject")["Condition"].nunique()
    complete = counts[counts == len(conditions)].index
    return long_df[long_df["Subject"].isin(complete)]


def run_omnibus_at_timepoint(
    long_df: pd.DataFrame, test_name: str, conditions: list[str]
) -> tuple[float, float, int]:
    """Run an omnibus test at a single timepoint.

    Args:
        long_df: DataFrame with columns [Subject, Condition, Value].
        test_name: one of 'rm-ANOVA', 'Linear Mixed Model', 'Friedman',
                   'One-way ANOVA', 'Kruskal-Wallis'.
        conditions: ordered list of condition names.

    Returns:
        (test_statistic, p_value, n_subjects_used)
    """
    if long_df.empty or long_df["Condition"].nunique() < 2:
        return np.nan, np.nan, 0

    groups = [
        long_df.loc[long_df["Condition"] == c, "Value"].values
        for c in conditions
        if c in long_df["Condition"].values
    ]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return np.nan, np.nan, 0

    n_subjects = long_df["Subject"].nunique()

    try:
        if test_name == "rm-ANOVA":
            comp = _complete_cases(long_df, conditions)
            n_complete = comp["Subject"].nunique()
            if n_complete < 3:
                return np.nan, np.nan, n_complete
            aovrm = AnovaRM(comp, "Value", "Subject", within=["Condition"])
            res = aovrm.fit()
            f_val = res.anova_table["F Value"].values[0]
            p_val = res.anova_table["Pr > F"].values[0]
            return f_val, p_val, n_complete

        elif test_name == "Linear Mixed Model":
            if n_subjects < 3:
                return np.nan, np.nan, n_subjects
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = smf.mixedlm(
                    "Value ~ C(Condition)", long_df, groups=long_df["Subject"]
                ).fit(reml=False)
                null = smf.mixedlm(
                    "Value ~ 1", long_df, groups=long_df["Subject"]
                ).fit(reml=False)
            lr_stat = -2 * (null.llf - full.llf)
            df_diff = len(conditions) - 1
            p_val = stats.chi2.sf(lr_stat, df_diff)
            return lr_stat, p_val, n_subjects

        elif test_name == "Friedman":
            comp = _complete_cases(long_df, conditions)
            n_complete = comp["Subject"].nunique()
            if n_complete < 3:
                return np.nan, np.nan, n_complete
            groups_c = [
                comp.loc[comp["Condition"] == c, "Value"]
                .sort_index()
                .values
                for c in conditions
            ]
            # Align by subject order
            subjects_sorted = sorted(comp["Subject"].unique())
            groups_c = []
            for c in conditions:
                sub_df = comp[comp["Condition"] == c].set_index("Subject")
                groups_c.append(sub_df.loc[subjects_sorted, "Value"].values)
            stat, p = stats.friedmanchisquare(*groups_c)
            return stat, p, n_complete

        elif test_name == "One-way ANOVA":
            stat, p = stats.f_oneway(*groups)
            return stat, p, n_subjects

        elif test_name == "Kruskal-Wallis":
            stat, p = stats.kruskal(*groups)
            return stat, p, n_subjects

    except Exception:
        return np.nan, np.nan, 0

    return np.nan, np.nan, 0


# Map omnibus test to the appropriate post-hoc pairwise test
_POSTHOC_TEST_MAP = {
    "rm-ANOVA": ("paired", "ttest"),
    "Linear Mixed Model": ("paired", "ttest"),
    "Friedman": ("paired", "wilcoxon"),
    "One-way ANOVA": ("independent", "ttest"),
    "Kruskal-Wallis": ("independent", "mannwhitney"),
}


def run_posthoc_at_timepoint(
    long_df: pd.DataFrame,
    omnibus_test: str,
    conditions: list[str],
    correction: str = "Bonferroni",
    alpha: float = 0.05,
) -> dict[tuple[str, str], dict]:
    """Run pairwise post-hoc comparisons with correction across pairs.

    The pairwise test is automatically chosen to match the omnibus test
    (parametric omnibus → parametric post-hoc, etc.).

    Returns:
        dict mapping (condA, condB) to
        {stat, p_raw, p_corrected, significant}
    """
    pair_design, pair_test = _POSTHOC_TEST_MAP.get(
        omnibus_test, ("independent", "ttest")
    )
    pairs = list(combinations(conditions, 2))
    if not pairs:
        return {}

    results = {}
    raw_p_values = []

    for ca, cb in pairs:
        data_a = long_df[long_df["Condition"] == ca]
        data_b = long_df[long_df["Condition"] == cb]

        if pair_design == "paired":
            shared = sorted(set(data_a["Subject"]) & set(data_b["Subject"]))
            if len(shared) < 2:
                raw_p_values.append(np.nan)
                results[(ca, cb)] = {"stat": np.nan, "p_raw": np.nan}
                continue
            a_vals = (
                data_a.set_index("Subject").loc[shared, "Value"].values.astype(float)
            )
            b_vals = (
                data_b.set_index("Subject").loc[shared, "Value"].values.astype(float)
            )
            if pair_test == "ttest":
                stat, p = stats.ttest_rel(a_vals, b_vals)
            else:  # wilcoxon
                if len(shared) < 6:
                    raw_p_values.append(np.nan)
                    results[(ca, cb)] = {"stat": np.nan, "p_raw": np.nan}
                    continue
                stat, p = stats.wilcoxon(a_vals, b_vals)
        else:  # independent
            a_vals = data_a["Value"].values.astype(float)
            b_vals = data_b["Value"].values.astype(float)
            if len(a_vals) < 2 or len(b_vals) < 2:
                raw_p_values.append(np.nan)
                results[(ca, cb)] = {"stat": np.nan, "p_raw": np.nan}
                continue
            if pair_test == "ttest":
                stat, p = stats.ttest_ind(a_vals, b_vals)
            else:  # mannwhitney
                stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")

        raw_p_values.append(p)
        results[(ca, cb)] = {"stat": stat, "p_raw": p}

    # Correct across pairs
    raw_p = np.array(raw_p_values)
    valid_mask = ~np.isnan(raw_p)
    corrected_p = np.full_like(raw_p, np.nan)

    if valid_mask.any():
        corr_map = {"Bonferroni": "bonferroni", "Holm-Bonferroni": "holm"}
        sm_method = corr_map.get(correction, "bonferroni")
        _, p_adj, _, _ = multipletests(
            raw_p[valid_mask], alpha=alpha, method=sm_method
        )
        corrected_p[valid_mask] = p_adj

    for idx, (ca, cb) in enumerate(pairs):
        results[(ca, cb)]["p_corrected"] = corrected_p[idx]
        results[(ca, cb)]["significant"] = (
            bool(corrected_p[idx] < alpha)
            if not np.isnan(corrected_p[idx])
            else False
        )

    return results


def run_omnibus_posthoc_across_timepoints(
    condition_dfs: dict[str, "pd.DataFrame"],
    timepoints: list[str],
    conditions: list[str],
    omnibus_test: str,
    timepoint_correction: str = "FDR (Benjamini-Hochberg)",
    posthoc_correction: str = "Bonferroni",
    alpha: float = 0.05,
) -> dict:
    """Run omnibus test + conditional post-hoc across all timepoints.

    Two correction levels (standard in repeated-measures time-series analysis):
    1. Omnibus p-values are corrected across timepoints.
    2. Post-hoc p-values are corrected across pairs *within* each
       significant timepoint (the omnibus already gates by timepoint).

    Args:
        condition_dfs: {condition_name: DataFrame} with index=timepoints,
            columns=subjects.
        timepoints: ordered list of timepoint labels to analyse.
        conditions: ordered list of condition names.
        omnibus_test: name of omnibus test.
        timepoint_correction: correction method across timepoints.
        posthoc_correction: correction method across pairs at each timepoint.
        alpha: significance threshold.

    Returns:
        dict with keys:
            'omnibus': dict with test_stats, p_values, p_corrected,
                       significant, n_subjects arrays.
            'posthoc': {(condA, condB): dict with test_stats, p_values,
                        p_corrected, significant arrays} — NaN where
                        omnibus was not significant.
    """
    from utils.data_loader import build_long_format_at_timepoint

    n_tp = len(timepoints)
    omnibus_stats = np.full(n_tp, np.nan)
    omnibus_p = np.full(n_tp, np.nan)
    omnibus_n = np.full(n_tp, 0, dtype=int)

    pairs = list(combinations(conditions, 2))
    posthoc = {
        pair: {
            "test_stats": np.full(n_tp, np.nan),
            "p_values": np.full(n_tp, np.nan),
            "p_corrected": np.full(n_tp, np.nan),
            "significant": np.full(n_tp, False),
        }
        for pair in pairs
    }

    # Cache long DataFrames (needed again for post-hoc)
    long_dfs = {}
    for t_idx, tp in enumerate(timepoints):
        long_df = build_long_format_at_timepoint(condition_dfs, tp)
        long_dfs[t_idx] = long_df
        if long_df.empty:
            continue
        stat, p, n = run_omnibus_at_timepoint(long_df, omnibus_test, conditions)
        omnibus_stats[t_idx] = stat
        omnibus_p[t_idx] = p
        omnibus_n[t_idx] = n

    # Correct omnibus p-values across timepoints
    omnibus_p_corr = apply_correction(omnibus_p, timepoint_correction, alpha)
    omnibus_sig = omnibus_p_corr < alpha

    # Post-hoc only at significant timepoints
    for t_idx in range(n_tp):
        if not omnibus_sig[t_idx]:
            continue
        long_df = long_dfs[t_idx]
        if long_df.empty:
            continue
        ph = run_posthoc_at_timepoint(
            long_df, omnibus_test, conditions, posthoc_correction, alpha
        )
        for pair in pairs:
            if pair in ph:
                posthoc[pair]["test_stats"][t_idx] = ph[pair]["stat"]
                posthoc[pair]["p_values"][t_idx] = ph[pair]["p_raw"]
                posthoc[pair]["p_corrected"][t_idx] = ph[pair]["p_corrected"]
                posthoc[pair]["significant"][t_idx] = ph[pair]["significant"]

    return {
        "omnibus": {
            "test_stats": omnibus_stats,
            "p_values": omnibus_p,
            "p_corrected": omnibus_p_corr,
            "significant": omnibus_sig,
            "n_subjects": omnibus_n,
            "test_name": omnibus_test,
        },
        "posthoc": posthoc,
    }
