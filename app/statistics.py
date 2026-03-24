"""Statistical tests and multiple comparisons correction."""

import numpy as np
from scipy import stats
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
