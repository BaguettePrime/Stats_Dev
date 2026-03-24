"""Excel parsing, condition detection, and data validation."""

import re
import pandas as pd
import numpy as np


def extract_condition_name(filename: str) -> str:
    """Extract condition letter from filename.

    Expects pattern like Cere_T_A_V_H.xlsx or Cere_T_A_V+H.xlsx,
    where the condition is the letter between the 2nd and 3rd underscore.
    Falls back to full filename stem if pattern doesn't match.
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return stem


def load_excel_file(uploaded_file) -> dict[str, pd.DataFrame]:
    """Load all sheets from an uploaded Excel file.

    Returns a dict mapping sheet name to DataFrame with:
    - Index: time window strings (e.g., '(-70, -60)')
    - Columns: subject identifiers (e.g., 'An1', 'An2')
    - Values: float data
    """
    xls = pd.ExcelFile(uploaded_file)
    sheets = {}
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        # First column is time window labels
        df = df.set_index(df.columns[0])
        df.index.name = "time_window"
        df.index = df.index.astype(str)
        # Ensure numeric data
        df = df.apply(pd.to_numeric, errors="coerce")
        sheets[sheet_name] = df
    return sheets


def find_common_sheets(all_sheets: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Find sheet names common to all loaded files."""
    if not all_sheets:
        return []
    sheet_sets = [set(s.keys()) for s in all_sheets.values()]
    common = sheet_sets[0]
    for s in sheet_sets[1:]:
        common = common & s
    return sorted(common)


def find_all_sheets(all_sheets: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Find all unique sheet names across all files."""
    if not all_sheets:
        return []
    all_names = set()
    for s in all_sheets.values():
        all_names.update(s.keys())
    return sorted(all_names)


def parse_time_window(label: str) -> tuple[float, float] | None:
    """Parse a time window string like '(-70, -60)' into (start, end)."""
    match = re.match(r"\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?", str(label))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def get_midpoints(df: pd.DataFrame) -> np.ndarray:
    """Compute midpoints of time windows from a DataFrame's index."""
    midpoints = []
    for label in df.index:
        parsed = parse_time_window(label)
        if parsed:
            midpoints.append((parsed[0] + parsed[1]) / 2.0)
        else:
            midpoints.append(np.nan)
    return np.array(midpoints)


def has_baseline(df: pd.DataFrame) -> bool:
    """Check if any time windows have negative start values (baseline)."""
    for label in df.index:
        parsed = parse_time_window(label)
        if parsed and parsed[0] < 0:
            return True
    return False


def detect_event_gap(df: pd.DataFrame) -> bool:
    """Detect if there's a gap around t=0 (baseline to post-event transition)."""
    midpoints = get_midpoints(df)
    has_neg = any(m < 0 for m in midpoints if not np.isnan(m))
    has_pos = any(m > 0 for m in midpoints if not np.isnan(m))
    return has_neg and has_pos


def get_valid_subjects(df: pd.DataFrame) -> list[str]:
    """Return subject columns that are not entirely NaN."""
    valid = []
    for col in df.columns:
        if not df[col].isna().all():
            valid.append(col)
    return valid


def detect_design(df_a: pd.DataFrame, df_b: pd.DataFrame, threshold: float = 0.8) -> dict:
    """Auto-detect experimental design (repeated measures vs independent).

    Returns dict with:
    - 'design': 'repeated' or 'independent'
    - 'shared_subjects': list of shared subject names
    - 'subjects_a': valid subjects in A
    - 'subjects_b': valid subjects in B
    - 'overlap_ratio': fraction of overlap
    """
    subjects_a = set(get_valid_subjects(df_a))
    subjects_b = set(get_valid_subjects(df_b))
    shared = subjects_a & subjects_b
    total = min(len(subjects_a), len(subjects_b))
    if total == 0:
        ratio = 0.0
    else:
        ratio = len(shared) / total

    return {
        "design": "repeated" if ratio >= threshold else "independent",
        "shared_subjects": sorted(shared),
        "subjects_a": sorted(subjects_a),
        "subjects_b": sorted(subjects_b),
        "overlap_ratio": ratio,
        "n_shared": len(shared),
        "n_a": len(subjects_a),
        "n_b": len(subjects_b),
    }


def get_shared_timepoints(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Return time window labels present in both DataFrames."""
    return [t for t in df_a.index if t in df_b.index]


def prepare_paired_data(
    df_a: pd.DataFrame, df_b: pd.DataFrame, subjects: list[str], timepoints: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare aligned data arrays for paired comparisons.

    Returns (data_a, data_b) each of shape (n_timepoints, n_subjects).
    Only includes subjects present in both and rows for shared timepoints.
    """
    data_a = df_a.loc[timepoints, subjects].values.astype(float)
    data_b = df_b.loc[timepoints, subjects].values.astype(float)
    return data_a, data_b


def prepare_independent_data(
    df_a: pd.DataFrame, df_b: pd.DataFrame, timepoints: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare data arrays for independent comparisons.

    Returns (data_a, data_b) which may have different numbers of columns.
    """
    subjects_a = get_valid_subjects(df_a)
    subjects_b = get_valid_subjects(df_b)
    data_a = df_a.loc[timepoints, subjects_a].values.astype(float)
    data_b = df_b.loc[timepoints, subjects_b].values.astype(float)
    return data_a, data_b


def detect_design_multi(
    condition_dfs: dict[str, pd.DataFrame], threshold: float = 0.8
) -> dict:
    """Detect experimental design across multiple (>2) conditions.

    Returns dict with design type, shared subjects, and overlap ratio.
    """
    all_valid = {c: set(get_valid_subjects(df)) for c, df in condition_dfs.items()}
    shared = set.intersection(*all_valid.values()) if all_valid else set()
    min_n = min((len(s) for s in all_valid.values()), default=0)
    ratio = len(shared) / min_n if min_n > 0 else 0.0

    return {
        "design": "repeated" if ratio >= threshold else "independent",
        "shared_subjects": sorted(shared),
        "overlap_ratio": ratio,
        "n_shared": len(shared),
        "n_per_condition": {c: len(s) for c, s in all_valid.items()},
    }


def get_common_timepoints_multi(
    condition_dfs: dict[str, pd.DataFrame],
) -> list[str]:
    """Return timepoints present in all condition DataFrames, preserving order."""
    sets = [set(df.index) for df in condition_dfs.values()]
    common = set.intersection(*sets) if sets else set()
    first_df = next(iter(condition_dfs.values()))
    return [t for t in first_df.index if t in common]


def build_long_format_all_timepoints(
    condition_dfs: dict[str, pd.DataFrame], timepoints: list[str]
) -> pd.DataFrame:
    """Build a long-format DataFrame across all timepoints.

    Returns DataFrame with columns [Subject, Condition, Time, Value].
    Rows with NaN values are dropped.
    """
    rows = []
    for cond_name, df in condition_dfs.items():
        for tp in timepoints:
            if tp not in df.index:
                continue
            for subject in get_valid_subjects(df):
                val = df.loc[tp, subject]
                if not np.isnan(val):
                    rows.append(
                        {
                            "Subject": subject,
                            "Condition": cond_name,
                            "Time": tp,
                            "Value": float(val),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["Subject", "Condition", "Time", "Value"])
    return pd.DataFrame(rows)


def build_long_format_at_timepoint(
    condition_dfs: dict[str, pd.DataFrame], timepoint: str
) -> pd.DataFrame:
    """Build a long-format DataFrame for a single timepoint.

    Returns DataFrame with columns [Subject, Condition, Value].
    Rows with NaN values are dropped.
    """
    rows = []
    for cond_name, df in condition_dfs.items():
        if timepoint not in df.index:
            continue
        for subject in get_valid_subjects(df):
            val = df.loc[timepoint, subject]
            if not np.isnan(val):
                rows.append(
                    {"Subject": subject, "Condition": cond_name, "Value": float(val)}
                )
    if not rows:
        return pd.DataFrame(columns=["Subject", "Condition", "Value"])
    return pd.DataFrame(rows)
