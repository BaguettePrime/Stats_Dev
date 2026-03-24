"""Main Streamlit entry point for time series analysis app."""

import re

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_loader import (
    extract_condition_name,
    load_excel_file,
    find_common_sheets,
    find_all_sheets,
    get_midpoints,
    get_valid_subjects,
    detect_design,
    detect_design_multi,
    get_shared_timepoints,
    get_common_timepoints_multi,
    prepare_paired_data,
    prepare_independent_data,
    has_baseline,
    detect_event_gap,
)
from utils.statistics import run_tests_across_timepoints, run_omnibus_posthoc_across_timepoints
from utils.plotting import create_figure, DEFAULT_COLORS
from utils.export import fig_to_svg, fig_to_pdf, fig_to_png, stats_to_csv, create_batch_zip

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("Time Series Analysis & Statistical Comparison")

# ── File Upload ──────────────────────────────────────────────────────────────
st.header("1. Upload Data")
uploaded_files = st.file_uploader(
    "Upload Excel files (one per condition)",
    type=["xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more .xlsx files to get started. Each file represents one experimental condition.")
    st.stop()

# ── Load Data ────────────────────────────────────────────────────────────────
# Cache data in session state to avoid re-parsing on every interaction
file_key = tuple(sorted(f.name for f in uploaded_files))
if st.session_state.get("_file_key") != file_key:
    all_sheets = {}
    condition_names_raw = {}
    for f in uploaded_files:
        cond = extract_condition_name(f.name)
        sheets = load_excel_file(f)
        all_sheets[cond] = sheets
        condition_names_raw[cond] = cond
    # Sort so _A_ conditions come first
    sorted_conditions = sorted(all_sheets.keys())
    st.session_state["all_sheets"] = all_sheets
    st.session_state["sorted_conditions"] = sorted_conditions
    st.session_state["condition_names_raw"] = condition_names_raw
    st.session_state["_file_key"] = file_key

all_sheets = st.session_state["all_sheets"]
sorted_conditions = st.session_state["sorted_conditions"]

# ── Condition Setup ──────────────────────────────────────────────────────────
st.header("2. Condition Setup")
col_setup = st.columns(len(sorted_conditions))
condition_display_names = {}
condition_order = []

for i, cond in enumerate(sorted_conditions):
    with col_setup[i]:
        display = st.text_input(
            f"Condition {cond} label",
            value=cond,
            key=f"cond_label_{cond}",
        )
        condition_display_names[cond] = display
        condition_order.append(cond)

st.caption(f"Conditions (in order): {', '.join(condition_display_names[c] for c in condition_order)}")

# ── Sheet Detection ──────────────────────────────────────────────────────────
common_sheets = find_common_sheets(all_sheets)
all_sheet_names = find_all_sheets(all_sheets)
sheets_only_in_some = [s for s in all_sheet_names if s not in common_sheets]

if not all_sheet_names:
    st.error("No sheets found in uploaded files.")
    st.stop()

# ── Sidebar Configuration ────────────────────────────────────────────────────
st.sidebar.header("Plot Configuration")

# Sheet selector
selected_sheets = st.sidebar.multiselect(
    "Sheets to plot",
    options=all_sheet_names,
    default=common_sheets[:1] if common_sheets else all_sheet_names[:1],
)

if not selected_sheets:
    st.warning("Select at least one sheet to plot.")
    st.stop()

# Plot type
plot_type = st.sidebar.radio(
    "Plot type",
    ["Line + shaded SEM", "Line + error bars"],
    index=0,
)

# Colors
st.sidebar.subheader("Colors")
colors = {}
for i, cond in enumerate(condition_order):
    default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    colors[cond] = st.sidebar.color_picker(
        f"{condition_display_names[cond]} color",
        value=default_color,
        key=f"color_{cond}",
    )

# Axis labels and title (per-sheet defaults set later)
with st.sidebar.expander("Labels & Fonts", expanded=False):
    custom_title = st.text_input("Title (leave blank for sheet name)", value="", key="custom_title")
    y_label = st.text_input("Y-axis label (leave blank for sheet name)", value="", key="y_label")
    x_label = st.text_input("X-axis label", value="Time (s)", key="x_label")
    font_size_title = st.slider("Title font size", 8, 28, 14, key="fs_title")
    font_size_axis = st.slider("Axis label font size", 6, 24, 12, key="fs_axis")
    font_size_tick = st.slider("Tick label font size", 6, 20, 10, key="fs_tick")
    font_size_legend = st.slider("Legend font size", 6, 20, 10, key="fs_legend")

with st.sidebar.expander("Figure Size", expanded=False):
    fig_width = st.slider("Width (inches)", 4.0, 20.0, 10.0, 0.5, key="fig_w")
    fig_height = st.slider("Height (inches)", 3.0, 15.0, 6.0, 0.5, key="fig_h")

legend_position = st.sidebar.selectbox(
    "Legend position",
    ["upper right", "upper left", "lower right", "lower left", "outside right"],
    index=0,
)

# ── Statistical Settings ─────────────────────────────────────────────────────
st.sidebar.header("Statistical Analysis")

# Build pairwise comparison options
if len(condition_order) >= 2:
    pair_options = []
    for i in range(len(condition_order)):
        for j in range(i + 1, len(condition_order)):
            ca, cb = condition_order[i], condition_order[j]
            pair_options.append((ca, cb))

    pair_labels = [
        f"{condition_display_names[a]} vs {condition_display_names[b]}"
        for a, b in pair_options
    ]

    selected_pair_labels = st.sidebar.multiselect(
        "Comparisons",
        options=pair_labels,
        default=pair_labels[:1],
    )
    selected_pairs = [pair_options[pair_labels.index(l)] for l in selected_pair_labels]
else:
    selected_pairs = []

# Detect design for each pair
designs = {}
if selected_pairs and common_sheets:
    ref_sheet = common_sheets[0]
    for ca, cb in selected_pairs:
        df_a = all_sheets[ca][ref_sheet]
        df_b = all_sheets[cb][ref_sheet]
        designs[(ca, cb)] = detect_design(df_a, df_b)

# Display design detection and allow override
design_overrides = {}
for (ca, cb), info in designs.items():
    label = f"{condition_display_names[ca]} vs {condition_display_names[cb]}"
    detected = info["design"]
    det_str = (
        f"Detected: **{'repeated measures' if detected == 'repeated' else 'independent'}** "
        f"({info['n_shared']}/{min(info['n_a'], info['n_b'])} subjects shared)"
    )
    st.sidebar.markdown(f"**{label}**: {det_str}")
    override = st.sidebar.checkbox(
        f"Override to {'independent' if detected == 'repeated' else 'repeated measures'} ({label})",
        value=False,
        key=f"override_{ca}_{cb}",
    )
    if override:
        design_overrides[(ca, cb)] = "independent" if detected == "repeated" else "repeated"
    else:
        design_overrides[(ca, cb)] = detected

# ── Analysis mode ────────────────────────────────────────────────────────────
analysis_mode = "Pairwise"
omnibus_test = None
posthoc_correction = "Bonferroni"
multi_design = "independent"

if len(condition_order) >= 3:
    analysis_mode = st.sidebar.radio(
        "Analysis mode",
        ["Pairwise", "Omnibus + Post-hoc"],
        index=0,
        help=(
            "**Pairwise**: test each pair independently across timepoints. "
            "**Omnibus + Post-hoc**: run an omnibus test first (ANOVA / LMM); "
            "post-hoc pairwise comparisons are only performed at timepoints "
            "where the omnibus is significant."
        ),
    )

if analysis_mode == "Omnibus + Post-hoc":
    # Detect multi-condition design
    if common_sheets:
        ref_sheet = common_sheets[0]
        multi_cond_dfs = {c: all_sheets[c][ref_sheet] for c in condition_order}
        multi_design_info = detect_design_multi(multi_cond_dfs)
    else:
        multi_design_info = {
            "design": "independent", "n_shared": 0,
            "overlap_ratio": 0.0, "n_per_condition": {},
        }

    multi_design = multi_design_info["design"]
    st.sidebar.markdown(
        f"**Design**: {'repeated measures' if multi_design == 'repeated' else 'independent'} "
        f"({multi_design_info['n_shared']} subjects shared across all conditions)"
    )

    override_multi = st.sidebar.checkbox(
        f"Override to {'independent' if multi_design == 'repeated' else 'repeated measures'}",
        value=False,
        key="override_multi_design",
    )
    if override_multi:
        multi_design = "independent" if multi_design == "repeated" else "repeated"

    # Omnibus test options depend on design
    if multi_design == "repeated":
        omnibus_options = ["rm-ANOVA", "Linear Mixed Model", "Friedman"]
        omnibus_help = (
            "**rm-ANOVA**: parametric, requires complete cases (all subjects in every condition). "
            "**Linear Mixed Model**: parametric, handles missing subjects naturally. "
            "**Friedman**: non-parametric, requires complete cases."
        )
    else:
        omnibus_options = ["One-way ANOVA", "Kruskal-Wallis"]
        omnibus_help = (
            "**One-way ANOVA**: parametric. "
            "**Kruskal-Wallis**: non-parametric."
        )

    omnibus_test = st.sidebar.selectbox(
        "Omnibus test", omnibus_options, index=0, help=omnibus_help
    )

    posthoc_correction = st.sidebar.selectbox(
        "Post-hoc correction (across pairs)",
        ["Bonferroni", "Holm-Bonferroni"],
        index=0,
        help="Correction applied across pairwise comparisons at each significant timepoint.",
    )

    correction_method = st.sidebar.selectbox(
        "Timepoint correction (omnibus p-values)",
        ["FDR (Benjamini-Hochberg)", "Bonferroni", "Holm-Bonferroni", "No correction"],
        index=0,
    )

    alpha = st.sidebar.number_input(
        "Significance threshold (\u03b1)", 0.001, 0.1, 0.05, 0.005
    )

    # Set test_name so pairwise code path is skipped
    test_name = "No statistics"

else:
    # ── Pairwise mode (original) ────────────────────────────────────────────
    test_options_repeated = [
        "Permutation test",
        "Paired t-test",
        "Wilcoxon signed-rank",
        "No statistics",
    ]
    test_options_independent = [
        "Permutation test",
        "Independent t-test",
        "Mann-Whitney U",
        "No statistics",
    ]

    any_repeated = any(d == "repeated" for d in design_overrides.values())
    any_independent = any(d == "independent" for d in design_overrides.values())

    if any_repeated and any_independent:
        all_test_options = list(dict.fromkeys(test_options_repeated + test_options_independent))
    elif any_repeated:
        all_test_options = test_options_repeated
    elif any_independent:
        all_test_options = test_options_independent
    else:
        all_test_options = ["No statistics"]

    test_name = st.sidebar.selectbox("Statistical test", all_test_options, index=0)

    correction_method = st.sidebar.selectbox(
        "Multiple comparisons correction",
        ["FDR (Benjamini-Hochberg)", "Bonferroni", "Holm-Bonferroni", "No correction"],
        index=0,
    )

    alpha = st.sidebar.number_input(
        "Significance threshold (\u03b1)", 0.001, 0.1, 0.05, 0.005
    )

with st.sidebar.expander("Significance display", expanded=False):
    use_multi_thresh = st.checkbox("Multiple significance levels", value=False)
    if use_multi_thresh:
        thresh2 = st.number_input("** threshold", 0.001, 0.1, 0.01, 0.001)
        thresh3 = st.number_input("*** threshold", 0.0001, 0.05, 0.001, 0.0001)
        sig_thresholds = [
            (alpha, f"p < {alpha}"),
            (thresh2, f"p < {thresh2}"),
            (thresh3, f"p < {thresh3}"),
        ]
    else:
        sig_thresholds = [(alpha, f"p < {alpha}")]

show_p_curve = st.sidebar.checkbox("Show p-value curve", value=False)

# Export settings
st.sidebar.header("Export")
export_dpi = st.sidebar.number_input("PNG DPI", 72, 600, 300, 50)

# ── Plot Generation ──────────────────────────────────────────────────────────
st.header("3. Results")

if sheets_only_in_some:
    st.caption(
        f"Sheets in only some files (stats unavailable): {', '.join(sheets_only_in_some)}"
    )


def _compute_mean_sem(df):
    """Compute mean and SEM across valid subjects."""
    valid = get_valid_subjects(df)
    values = df[valid].values.astype(float)
    mean = np.nanmean(values, axis=1)
    n = np.sum(~np.isnan(values), axis=1)
    sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(n)
    return mean, sem


def _timepoints_to_midpoints(timepoints):
    """Convert timepoint labels to midpoint floats."""
    midpts = []
    for tp in timepoints:
        match = re.match(r"\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?", tp)
        if match:
            midpts.append((float(match.group(1)) + float(match.group(2))) / 2)
        else:
            midpts.append(np.nan)
    return np.array(midpts)


def generate_plot_and_stats(sheet_name):
    """Generate figure and stats for a single sheet. Returns (fig, stats_results, stats_csvs, omnibus_info)."""
    # Which conditions have this sheet
    conds_with_sheet = [c for c in condition_order if sheet_name in all_sheets[c]]

    # Build condition_data for plotting
    condition_data = {}
    for cond in conds_with_sheet:
        condition_data[condition_display_names[cond]] = {
            "df": all_sheets[cond][sheet_name],
            "color": colors[cond],
        }

    # Stats
    stats_results = {}
    stats_csv_data = {}
    omnibus_info = None

    # ── Omnibus + Post-hoc path ─────────────────────────────────────────────
    if (
        analysis_mode == "Omnibus + Post-hoc"
        and omnibus_test is not None
        and len(conds_with_sheet) >= 3
        and sheet_name in common_sheets
    ):
        cond_dfs = {c: all_sheets[c][sheet_name] for c in conds_with_sheet}
        shared_tp = get_common_timepoints_multi(cond_dfs)

        if shared_tp:
            result = run_omnibus_posthoc_across_timepoints(
                condition_dfs=cond_dfs,
                timepoints=shared_tp,
                conditions=conds_with_sheet,
                omnibus_test=omnibus_test,
                timepoint_correction=correction_method,
                posthoc_correction=posthoc_correction,
                alpha=alpha,
            )

            midpts = _timepoints_to_midpoints(shared_tp)

            # Store omnibus info for display
            omnibus_info = result["omnibus"]
            omnibus_info["midpoints"] = midpts
            omnibus_info["timepoints"] = shared_tp

            # Convert post-hoc results to plotting format
            for (ca, cb), ph in result["posthoc"].items():
                comp_label = (
                    f"{condition_display_names[ca]} vs {condition_display_names[cb]}"
                )
                stats_results[comp_label] = {
                    "test_stats": ph["test_stats"],
                    "p_values": ph["p_values"],
                    "p_corrected": ph["p_corrected"],
                    "significant": ph["significant"],
                    "midpoints": midpts,
                }

    # ── Pairwise path (original) ────────────────────────────────────────────
    elif test_name != "No statistics" and len(conds_with_sheet) >= 2:
        for ca, cb in selected_pairs:
            if ca not in conds_with_sheet or cb not in conds_with_sheet:
                continue

            df_a = all_sheets[ca][sheet_name]
            df_b = all_sheets[cb][sheet_name]
            design = design_overrides.get((ca, cb), "independent")
            shared_tp = get_shared_timepoints(df_a, df_b)

            if not shared_tp:
                continue

            # Determine the right test for this pair's design
            pair_test = test_name
            if design == "repeated" and test_name == "Independent t-test":
                pair_test = "Paired t-test"
            elif design == "repeated" and test_name == "Mann-Whitney U":
                pair_test = "Wilcoxon signed-rank"
            elif design == "independent" and test_name == "Paired t-test":
                pair_test = "Independent t-test"
            elif design == "independent" and test_name == "Wilcoxon signed-rank":
                pair_test = "Mann-Whitney U"

            if design == "repeated":
                design_info = designs.get((ca, cb), detect_design(df_a, df_b))
                subjects = design_info["shared_subjects"]
                # Filter to subjects valid in both
                subjects = [
                    s for s in subjects
                    if s in get_valid_subjects(df_a) and s in get_valid_subjects(df_b)
                ]
                if len(subjects) < 2:
                    continue
                data_a, data_b = prepare_paired_data(df_a, df_b, subjects, shared_tp)
            else:
                data_a, data_b = prepare_independent_data(df_a, df_b, shared_tp)

            sr = run_tests_across_timepoints(
                data_a, data_b, pair_test, design, correction_method, alpha
            )

            midpts = _timepoints_to_midpoints(shared_tp)
            sr["midpoints"] = midpts

            comp_label = f"{condition_display_names[ca]} vs {condition_display_names[cb]}"
            stats_results[comp_label] = sr

            # Build CSV data
            mean_a, sem_a = _compute_mean_sem(df_a.loc[shared_tp])
            mean_b, sem_b = _compute_mean_sem(df_b.loc[shared_tp])
            csv_bytes = stats_to_csv(
                shared_tp, midpts,
                mean_a, sem_a, mean_b, sem_b,
                condition_display_names[ca], condition_display_names[cb],
                sr,
            )
            stats_csv_data[f"{sheet_name}_{comp_label}"] = csv_bytes

    config = {
        "plot_type": "error_bars" if plot_type == "Line + error bars" else "shaded_sem",
        "title": custom_title if custom_title else sheet_name,
        "y_label": y_label if y_label else sheet_name,
        "x_label": x_label,
        "font_size_title": font_size_title,
        "font_size_axis": font_size_axis,
        "font_size_tick": font_size_tick,
        "font_size_legend": font_size_legend,
        "fig_width": fig_width,
        "fig_height": fig_height,
        "legend_position": legend_position,
        "show_event_line": True,
        "show_p_curve": show_p_curve,
        "sig_thresholds": sig_thresholds,
    }

    fig = create_figure(condition_data, sheet_name, config, stats_results or None)
    return fig, stats_results, stats_csv_data, omnibus_info


# Generate plots for each selected sheet
all_figures = {}
all_stats_csvs = {}

for sheet_name in selected_sheets:
    st.subheader(f"Sheet: {sheet_name}")

    try:
        fig, stats_results, stats_csv_data, omnibus_info = generate_plot_and_stats(sheet_name)
        all_figures[sheet_name] = fig
        all_stats_csvs.update(stats_csv_data)

        st.pyplot(fig)
        plt.close(fig)

        # Display omnibus summary (Omnibus + Post-hoc mode)
        if omnibus_info is not None:
            with st.expander(f"Omnibus results \u2014 {sheet_name}", expanded=False):
                n_sig = int(np.sum(omnibus_info["significant"]))
                n_total = len(omnibus_info["significant"])
                st.write(
                    f"**{omnibus_info['test_name']}**: {n_sig}/{n_total} "
                    f"timepoints significant (\u03b1 = {alpha}, {correction_method})"
                )

                omnibus_df = pd.DataFrame({
                    "Midpoint": omnibus_info["midpoints"],
                    "Statistic": omnibus_info["test_stats"],
                    "p (raw)": omnibus_info["p_values"],
                    "p (corrected)": omnibus_info["p_corrected"],
                    "Significant": omnibus_info["significant"],
                    "N subjects": omnibus_info["n_subjects"],
                })
                st.dataframe(omnibus_df, use_container_width=True, hide_index=True)

                if n_sig > 0 and stats_results:
                    st.markdown("---")
                    st.write(
                        f"**Post-hoc pairwise tests** "
                        f"(correction across pairs: {posthoc_correction})"
                    )
                    for comp_label, sr in stats_results.items():
                        ph_sig = int(np.sum(sr["significant"]))
                        st.write(f"*{comp_label}*: {ph_sig} timepoints significant")
                        ph_df = pd.DataFrame({
                            "Midpoint": sr["midpoints"],
                            "p (raw)": sr["p_values"],
                            "p (corrected)": sr["p_corrected"],
                            "Significant": sr["significant"],
                        })
                        # Only show timepoints where omnibus was significant
                        mask = omnibus_info["significant"]
                        st.dataframe(
                            ph_df[mask].reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True,
                        )

        # Display pairwise stats summary (Pairwise mode)
        elif stats_results:
            with st.expander(f"Statistics details \u2014 {sheet_name}", expanded=False):
                for comp_label, sr in stats_results.items():
                    n_sig = np.sum(sr["significant"])
                    n_total = len(sr["significant"])
                    st.write(
                        f"**{comp_label}**: {n_sig}/{n_total} timepoints significant "
                        f"(\u03b1 = {alpha}, {correction_method})"
                    )
                    # Show a small table
                    midpts = sr["midpoints"]
                    summary_df = pd.DataFrame({
                        "Midpoint": midpts,
                        "p (raw)": sr["p_values"],
                        "p (corrected)": sr["p_corrected"],
                        "Significant": sr["significant"],
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Export buttons
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)

        # Re-create fig for export (since we closed it)
        fig_export, _, _, _ = generate_plot_and_stats(sheet_name)

        with col_e1:
            st.download_button(
                "Download SVG",
                fig_to_svg(fig_export),
                f"{sheet_name}.svg",
                "image/svg+xml",
                key=f"svg_{sheet_name}",
            )
        with col_e2:
            st.download_button(
                "Download PDF",
                fig_to_pdf(fig_export),
                f"{sheet_name}.pdf",
                "application/pdf",
                key=f"pdf_{sheet_name}",
            )
        with col_e3:
            st.download_button(
                "Download PNG",
                fig_to_png(fig_export, export_dpi),
                f"{sheet_name}.png",
                "image/png",
                key=f"png_{sheet_name}",
            )
        plt.close(fig_export)

        # Stats CSV downloads
        for csv_name, csv_bytes in stats_csv_data.items():
            with col_e4:
                st.download_button(
                    "Stats CSV",
                    csv_bytes,
                    f"{csv_name}.csv",
                    "text/csv",
                    key=f"csv_{csv_name}_{sheet_name}",
                )

    except Exception as e:
        st.error(f"Error plotting {sheet_name}: {e}")
        import traceback
        st.code(traceback.format_exc())

# ── Batch Export ─────────────────────────────────────────────────────────────
if len(selected_sheets) > 1:
    st.header("4. Batch Export")
    batch_fmt = st.selectbox("Batch export format", ["svg", "pdf", "png"])

    if st.button("Generate batch ZIP"):
        with st.spinner("Generating all figures..."):
            batch_figs = {}
            batch_csvs = {}
            for sn in selected_sheets:
                try:
                    fig_b, _, csv_data, _ = generate_plot_and_stats(sn)
                    batch_figs[sn] = fig_b
                    batch_csvs.update(csv_data)
                except Exception:
                    pass

            zip_bytes = create_batch_zip(batch_figs, batch_csvs, batch_fmt, export_dpi)
            for f in batch_figs.values():
                plt.close(f)

            st.download_button(
                "Download ZIP",
                zip_bytes,
                "batch_export.zip",
                "application/zip",
                key="batch_zip",
            )
