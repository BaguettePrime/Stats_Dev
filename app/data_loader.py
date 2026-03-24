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
