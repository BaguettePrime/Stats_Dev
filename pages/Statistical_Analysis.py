"""Page 2 – Statistical Analysis: pairwise, two-way ANOVA/LMM, omnibus + post-hoc."""

import re

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.shared_ui import (
    render_upload,
    render_condition_setup,
    get_sheet_info,
    get_excluded_subjects,
    filter_excluded_from_sheets,
    render_sidebar_colors,
    render_sidebar_plot_config,
)
from utils.data_loader import (
    get_midpoints,
    get_valid_subjects,
    detect_design,
    detect_design_multi,
    get_shared_timepoints,
    get_common_timepoints_multi,
    prepare_paired_data,
    prepare_independent_data,
    build_long_format_all_timepoints,
    detect_event_gap,
    find_common_sheets,
)
from utils.statistics import (
    run_tests_across_timepoints,
    run_omnibus_posthoc_across_timepoints,
    run_twoway_rm_anova,
    run_twoway_lmm,
)
from utils.plotting import create_figure, DEFAULT_COLORS
from utils.export import fig_to_svg, fig_to_pdf, fig_to_png, stats_to_csv, create_batch_zip

st.title("Statistical Analysis")

# ── Upload & Setup ───────────────────────────────────────────────────────────
st.header("1. Upload Data")
if not render_upload():
    st.stop()

st.header("2. Condition Setup")
display_names, condition_order = render_condition_setup()

all_sheets_raw, common_sheets, all_sheet_names = get_sheet_info()
sheets_only_in_some = [s for s in all_sheet_names if s not in common_sheets]

if not all_sheet_names:
    st.error("No sheets found in uploaded files.")
    st.stop()

# ── Apply subject exclusions ─────────────────────────────────────────────────
excluded = get_excluded_subjects()
all_sheets = filter_excluded_from_sheets(all_sheets_raw, excluded)

if excluded:
    st.caption(
        f"Excluded subjects (set on Data Explorer page): "
        f"{', '.join(sorted(excluded))}"
    )

common_sheets = find_common_sheets(all_sheets)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Plot Configuration")

selected_sheets = st.sidebar.multiselect(
    "Sheets to plot",
    options=all_sheet_names,
    default=common_sheets[:1] if common_sheets else all_sheet_names[:1],
    key="stats_sheets",
)
if not selected_sheets:
    st.warning("Select at least one sheet to plot.")
    st.stop()

colors = render_sidebar_colors(condition_order, display_names)
config = render_sidebar_plot_config()

# ── Statistical Settings ─────────────────────────────────────────────────────
st.sidebar.header("Statistical Analysis")

# Build pairwise comparison options
if len(condition_order) >= 2:
    pair_options = []
    for i in range(len(condition_order)):
        for j in range(i + 1, len(condition_order)):
            pair_options.append((condition_order[i], condition_order[j]))

    pair_labels = [
        f"{display_names[a]} vs {display_names[b]}" for a, b in pair_options
    ]
    selected_pair_labels = st.sidebar.multiselect(
        "Comparisons", pair_labels, default=pair_labels[:1],
    )
    selected_pairs = [pair_options[pair_labels.index(l)] for l in selected_pair_labels]
else:
    selected_pairs = []

# Design detection
designs = {}
if selected_pairs and common_sheets:
    ref = common_sheets[0]
    for ca, cb in selected_pairs:
        designs[(ca, cb)] = detect_design(all_sheets[ca][ref], all_sheets[cb][ref])

design_overrides = {}
for (ca, cb), info in designs.items():
    label = f"{display_names[ca]} vs {display_names[cb]}"
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
    design_overrides[(ca, cb)] = (
        ("independent" if detected == "repeated" else "repeated") if override else detected
    )

# ── Analysis mode ────────────────────────────────────────────────────────────
analysis_mode = "Pairwise"
omnibus_test = None
posthoc_correction = "Bonferroni"
multi_design = "independent"
twoway_test = None
twoway_followup_correction = "FDR (Benjamini-Hochberg)"

if len(condition_order) >= 2:
    mode_options = ["Pairwise", "Two-way (Condition × Time)"]
    if len(condition_order) >= 3:
        mode_options.append("Omnibus + Post-hoc")
    analysis_mode = st.sidebar.radio(
        "Analysis mode",
        mode_options,
        index=0,
        help=(
            "**Pairwise**: test each selected pair at each timepoint independently. "
            "Choose this when you have specific pairs to compare.\n\n"
            "**Two-way (Condition × Time)**: a single global test (rm-ANOVA or LMM) "
            "that tests whether there is a main effect of Condition, a main effect "
            "of Time, and crucially a Condition × Time *interaction* (i.e. do "
            "conditions diverge over time?). If the interaction is significant, "
            "follow-up pairwise tests at each timepoint show *where* the selected "
            "pairs differ.\n\n"
            "**Omnibus + Post-hoc** (≥ 3 conditions): at each timepoint, first runs "
            "a single omnibus test across *all* conditions (ANOVA / Kruskal-Wallis / "
            "Friedman). Only at timepoints where the omnibus is significant, "
            "post-hoc pairwise tests identify *which* pairs differ. This controls "
            "the family-wise error rate better than running all pairwise tests."
        ),
    )

# ── Two-way config ───────────────────────────────────────────────────────────
if analysis_mode == "Two-way (Condition × Time)":
    if common_sheets:
        ref = common_sheets[0]
        multi_dfs = {c: all_sheets[c][ref] for c in condition_order}
        mdi = detect_design_multi(multi_dfs)
    else:
        mdi = {"design": "independent", "n_shared": 0, "overlap_ratio": 0.0, "n_per_condition": {}}

    multi_design = mdi["design"]
    st.sidebar.markdown(
        f"**Design**: {'repeated measures' if multi_design == 'repeated' else 'independent'} "
        f"({mdi['n_shared']} subjects shared across all conditions)"
    )
    override_multi = st.sidebar.checkbox(
        f"Override to {'independent' if multi_design == 'repeated' else 'repeated measures'}",
        value=False, key="override_twoway_design",
    )
    if override_multi:
        multi_design = "independent" if multi_design == "repeated" else "repeated"

    if multi_design == "repeated":
        twoway_options = ["Two-way rm-ANOVA", "Two-way LMM"]
    else:
        twoway_options = ["Two-way LMM"]

    twoway_test = st.sidebar.selectbox("Two-way test", twoway_options, index=0)
    twoway_followup_correction = st.sidebar.selectbox(
        "Follow-up correction (per-timepoint)",
        ["FDR (Benjamini-Hochberg)", "Bonferroni", "Holm-Bonferroni", "No correction"],
        index=0,
    )
    alpha = st.sidebar.number_input("Significance threshold (α)", 0.001, 0.1, 0.05, 0.005)
    correction_method = twoway_followup_correction
    test_name = "No statistics"

# ── Omnibus + Post-hoc config ────────────────────────────────────────────────
elif analysis_mode == "Omnibus + Post-hoc":
    if common_sheets:
        ref = common_sheets[0]
        multi_dfs = {c: all_sheets[c][ref] for c in condition_order}
        mdi = detect_design_multi(multi_dfs)
    else:
        mdi = {"design": "independent", "n_shared": 0, "overlap_ratio": 0.0, "n_per_condition": {}}

    multi_design = mdi["design"]
    st.sidebar.markdown(
        f"**Design**: {'repeated measures' if multi_design == 'repeated' else 'independent'} "
        f"({mdi['n_shared']} subjects shared)"
    )
    override_multi = st.sidebar.checkbox(
        f"Override to {'independent' if multi_design == 'repeated' else 'repeated measures'}",
        value=False, key="override_multi_design",
    )
    if override_multi:
        multi_design = "independent" if multi_design == "repeated" else "repeated"

    if multi_design == "repeated":
        omnibus_options = ["rm-ANOVA", "Linear Mixed Model", "Friedman"]
    else:
        omnibus_options = ["One-way ANOVA", "Kruskal-Wallis"]

    omnibus_test = st.sidebar.selectbox("Omnibus test", omnibus_options, index=0)
    posthoc_correction = st.sidebar.selectbox(
        "Post-hoc correction (across pairs)", ["Bonferroni", "Holm-Bonferroni"], index=0,
    )
    correction_method = st.sidebar.selectbox(
        "Timepoint correction (omnibus p-values)",
        ["FDR (Benjamini-Hochberg)", "Bonferroni", "Holm-Bonferroni", "No correction"],
        index=0,
    )
    alpha = st.sidebar.number_input("Significance threshold (α)", 0.001, 0.1, 0.05, 0.005)
    test_name = "No statistics"

# ── Pairwise config ──────────────────────────────────────────────────────────
else:
    test_options_repeated = ["Permutation test", "Paired t-test", "Wilcoxon signed-rank", "No statistics"]
    test_options_independent = ["Permutation test", "Independent t-test", "Mann-Whitney U", "No statistics"]

    any_rep = any(d == "repeated" for d in design_overrides.values())
    any_ind = any(d == "independent" for d in design_overrides.values())
    if any_rep and any_ind:
        opts = list(dict.fromkeys(test_options_repeated + test_options_independent))
    elif any_rep:
        opts = test_options_repeated
    elif any_ind:
        opts = test_options_independent
    else:
        opts = ["No statistics"]

    test_name = st.sidebar.selectbox("Statistical test", opts, index=0)
    correction_method = st.sidebar.selectbox(
        "Multiple comparisons correction",
        ["FDR (Benjamini-Hochberg)", "Bonferroni", "Holm-Bonferroni", "No correction"],
        index=0,
    )
    alpha = st.sidebar.number_input("Significance threshold (α)", 0.001, 0.1, 0.05, 0.005)

# ── Shared display options ───────────────────────────────────────────────────
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

st.sidebar.header("Export")
export_dpi = st.sidebar.number_input("PNG DPI", 72, 600, 300, 50, key="stats_dpi")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_mean_sem(df):
    valid = get_valid_subjects(df)
    values = df[valid].values.astype(float)
    mean = np.nanmean(values, axis=1)
    n = np.sum(~np.isnan(values), axis=1)
    sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, sem


def _tp_to_mid(timepoints):
    midpts = []
    for tp in timepoints:
        m = re.match(r"\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?", tp)
        midpts.append((float(m.group(1)) + float(m.group(2))) / 2 if m else np.nan)
    return np.array(midpts)


# ── Plot Generation ──────────────────────────────────────────────────────────

def generate_plot_and_stats(sheet_name):
    """Generate figure + stats for one sheet.

    Returns (fig, stats_results, stats_csv_data, omnibus_info, twoway_info).
    """
    conds_with_sheet = [c for c in condition_order if sheet_name in all_sheets[c]]

    condition_data = {}
    for cond in conds_with_sheet:
        condition_data[display_names[cond]] = {
            "df": all_sheets[cond][sheet_name],
            "color": colors[cond],
        }

    stats_results = {}
    stats_csv_data = {}
    omnibus_info = None
    twoway_info = None

    # ── Two-way (Condition × Time) ──────────────────────────────────────
    if (
        analysis_mode == "Two-way (Condition × Time)"
        and twoway_test is not None
        and len(conds_with_sheet) >= 2
        and sheet_name in common_sheets
    ):
        cond_dfs = {c: all_sheets[c][sheet_name] for c in conds_with_sheet}
        shared_tp = get_common_timepoints_multi(cond_dfs)

        if shared_tp:
            long_df = build_long_format_all_timepoints(cond_dfs, shared_tp)

            if twoway_test == "Two-way rm-ANOVA":
                table, n_subj = run_twoway_rm_anova(long_df, conds_with_sheet, shared_tp)
            else:
                table, n_subj = run_twoway_lmm(long_df, conds_with_sheet, shared_tp)

            midpts = _tp_to_mid(shared_tp)
            twoway_info = {"table": table, "n_subjects": n_subj, "test_name": twoway_test}

            if table is not None:
                inter_p = table.get("Condition:Time", {}).get("p", 1.0)
                if inter_p < alpha:
                    # Only run follow-ups for user-selected pairs
                    followup_pairs = [
                        (ca, cb) for ca, cb in selected_pairs
                        if ca in conds_with_sheet and cb in conds_with_sheet
                    ]
                    for ca, cb in followup_pairs:
                        df_a = all_sheets[ca][sheet_name]
                        df_b = all_sheets[cb][sheet_name]
                        if multi_design == "repeated":
                            di = detect_design(df_a, df_b)
                            subjs = [
                                s for s in di["shared_subjects"]
                                if s in get_valid_subjects(df_a)
                                and s in get_valid_subjects(df_b)
                            ]
                            if len(subjs) < 2:
                                continue
                            da, db = prepare_paired_data(df_a, df_b, subjs, shared_tp)
                            pt, des = "Paired t-test", "repeated"
                        else:
                            da, db = prepare_independent_data(df_a, df_b, shared_tp)
                            pt, des = "Independent t-test", "independent"

                        sr = run_tests_across_timepoints(
                            da, db, pt, des, twoway_followup_correction, alpha,
                        )
                        sr["midpoints"] = midpts
                        stats_results[
                            f"{display_names[ca]} vs {display_names[cb]}"
                        ] = sr

    # ── Omnibus + Post-hoc ──────────────────────────────────────────────
    elif (
        analysis_mode == "Omnibus + Post-hoc"
        and omnibus_test is not None
        and len(conds_with_sheet) >= 3
        and sheet_name in common_sheets
    ):
        cond_dfs = {c: all_sheets[c][sheet_name] for c in conds_with_sheet}
        shared_tp = get_common_timepoints_multi(cond_dfs)
        if shared_tp:
            result = run_omnibus_posthoc_across_timepoints(
                cond_dfs, shared_tp, conds_with_sheet, omnibus_test,
                correction_method, posthoc_correction, alpha,
            )
            midpts = _tp_to_mid(shared_tp)
            omnibus_info = result["omnibus"]
            omnibus_info["midpoints"] = midpts
            omnibus_info["timepoints"] = shared_tp

            # Only show post-hoc results for user-selected pairs
            selected_set = set(selected_pairs)
            for (ca, cb), ph in result["posthoc"].items():
                if (ca, cb) not in selected_set and (cb, ca) not in selected_set:
                    continue
                stats_results[f"{display_names[ca]} vs {display_names[cb]}"] = {
                    "test_stats": ph["test_stats"],
                    "p_values": ph["p_values"],
                    "p_corrected": ph["p_corrected"],
                    "significant": ph["significant"],
                    "midpoints": midpts,
                }

    # ── Pairwise ────────────────────────────────────────────────────────
    elif test_name != "No statistics" and len(conds_with_sheet) >= 2:
        for ca, cb in selected_pairs:
            if ca not in conds_with_sheet or cb not in conds_with_sheet:
                continue
            df_a, df_b = all_sheets[ca][sheet_name], all_sheets[cb][sheet_name]
            design = design_overrides.get((ca, cb), "independent")
            shared_tp = get_shared_timepoints(df_a, df_b)
            if not shared_tp:
                continue

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
                di = designs.get((ca, cb), detect_design(df_a, df_b))
                subjs = [
                    s for s in di["shared_subjects"]
                    if s in get_valid_subjects(df_a) and s in get_valid_subjects(df_b)
                ]
                if len(subjs) < 2:
                    continue
                da, db = prepare_paired_data(df_a, df_b, subjs, shared_tp)
            else:
                da, db = prepare_independent_data(df_a, df_b, shared_tp)

            sr = run_tests_across_timepoints(
                da, db, pair_test, design, correction_method, alpha,
            )
            midpts = _tp_to_mid(shared_tp)
            sr["midpoints"] = midpts
            comp_label = f"{display_names[ca]} vs {display_names[cb]}"
            stats_results[comp_label] = sr

            mean_a, sem_a = _compute_mean_sem(df_a.loc[shared_tp])
            mean_b, sem_b = _compute_mean_sem(df_b.loc[shared_tp])
            csv_bytes = stats_to_csv(
                shared_tp, midpts, mean_a, sem_a, mean_b, sem_b,
                display_names[ca], display_names[cb], sr,
            )
            stats_csv_data[f"{sheet_name}_{comp_label}"] = csv_bytes

    plot_cfg = {
        "plot_type": config["plot_type"],
        "title": config["custom_title"] if config["custom_title"] else sheet_name,
        "y_label": config["y_label"] if config["y_label"] else sheet_name,
        "x_label": config["x_label"],
        "font_size_title": config["font_size_title"],
        "font_size_axis": config["font_size_axis"],
        "font_size_tick": config["font_size_tick"],
        "font_size_legend": config["font_size_legend"],
        "fig_width": config["fig_width"],
        "fig_height": config["fig_height"],
        "legend_position": config["legend_position"],
        "show_event_line": True,
        "show_p_curve": show_p_curve,
        "sig_thresholds": sig_thresholds,
    }
    fig = create_figure(condition_data, sheet_name, plot_cfg, stats_results or None)
    return fig, stats_results, stats_csv_data, omnibus_info, twoway_info


# ── Results ──────────────────────────────────────────────────────────────────
st.header("3. Results")

if sheets_only_in_some:
    st.caption(f"Sheets in only some files: {', '.join(sheets_only_in_some)}")

all_figures = {}
all_stats_csvs = {}

for sheet_name in selected_sheets:
    st.subheader(f"Sheet: {sheet_name}")

    try:
        fig, stats_results, stats_csv_data, omnibus_info, twoway_info = (
            generate_plot_and_stats(sheet_name)
        )
        all_figures[sheet_name] = fig
        all_stats_csvs.update(stats_csv_data)

        # Cache export bytes before closing
        svg_bytes = fig_to_svg(fig)
        pdf_bytes = fig_to_pdf(fig)
        png_bytes = fig_to_png(fig, export_dpi)

        st.pyplot(fig)
        plt.close(fig)

        # ── Two-way results ─────────────────────────────────────────────
        if twoway_info is not None and twoway_info["table"] is not None:
            with st.expander(f"Two-way results — {sheet_name}", expanded=True):
                tbl = twoway_info["table"]
                st.write(
                    f"**{twoway_info['test_name']}** "
                    f"(N = {twoway_info['n_subjects']} subjects)"
                )

                is_lmm = "LMM" in twoway_info["test_name"]
                stat_col = "LR stat" if is_lmm else "F"
                rows_data = []
                for effect, vals in tbl.items():
                    stat_val = vals.get("LR", vals.get("F", np.nan))
                    p_val = vals["p"]
                    sig = "✓" if p_val < alpha else ""
                    row = {
                        "Effect": effect,
                        stat_col: f"{stat_val:.3f}",
                        "p-value": f"{p_val:.4g}",
                    }
                    if is_lmm:
                        row["df"] = int(vals["df"])
                    else:
                        row["df (num)"] = int(vals["df_num"])
                        row["df (den)"] = int(vals["df_den"])
                    row[f"Sig. (p < {alpha})"] = sig
                    rows_data.append(row)
                st.table(pd.DataFrame(rows_data))

                inter_p = tbl.get("Condition:Time", {}).get("p", 1.0)
                if inter_p < alpha:
                    st.success(
                        f"Condition × Time interaction is significant "
                        f"(p = {inter_p:.4g}). Per-timepoint follow-up tests "
                        f"shown below (correction: {twoway_followup_correction})."
                    )
                    if stats_results:
                        for comp_label, sr in stats_results.items():
                            n_sig = int(np.sum(sr["significant"]))
                            n_total = len(sr["significant"])
                            st.write(
                                f"**{comp_label}**: {n_sig}/{n_total} "
                                f"timepoints significant"
                            )
                            detail_df = pd.DataFrame({
                                "Midpoint": sr["midpoints"],
                                "Test statistic": sr["test_stats"],
                                "p (raw)": sr["p_values"],
                                "p (corrected)": sr["p_corrected"],
                                "Significant": sr["significant"],
                            })
                            st.dataframe(
                                detail_df, use_container_width=True, hide_index=True,
                            )
                else:
                    st.info(
                        f"Condition × Time interaction is not significant "
                        f"(p = {inter_p:.4g}). No per-timepoint follow-up needed."
                    )

        elif twoway_info is not None and twoway_info["table"] is None:
            st.warning(
                f"Two-way analysis failed (only {twoway_info['n_subjects']} "
                f"complete subjects; need ≥3)."
            )

        # ── Omnibus + Post-hoc results ──────────────────────────────────
        elif omnibus_info is not None:
            with st.expander(f"Omnibus results — {sheet_name}", expanded=False):
                n_sig = int(np.sum(omnibus_info["significant"]))
                n_total = len(omnibus_info["significant"])
                st.write(
                    f"**{omnibus_info['test_name']}**: {n_sig}/{n_total} "
                    f"timepoints significant (α = {alpha}, {correction_method})"
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
                        mask = omnibus_info["significant"]
                        st.dataframe(
                            ph_df[mask].reset_index(drop=True),
                            use_container_width=True, hide_index=True,
                        )

        # ── Pairwise results ────────────────────────────────────────────
        elif stats_results:
            with st.expander(f"Statistics details — {sheet_name}", expanded=False):
                for comp_label, sr in stats_results.items():
                    n_sig = int(np.sum(sr["significant"]))
                    n_total = len(sr["significant"])
                    st.write(
                        f"**{comp_label}**: {n_sig}/{n_total} timepoints significant "
                        f"(α = {alpha}, {correction_method})"
                    )
                    summary_df = pd.DataFrame({
                        "Midpoint": sr["midpoints"],
                        "p (raw)": sr["p_values"],
                        "p (corrected)": sr["p_corrected"],
                        "Significant": sr["significant"],
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # ── Export buttons ──────────────────────────────────────────────
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            st.download_button(
                "Download SVG", svg_bytes,
                f"{sheet_name}.svg", "image/svg+xml",
                key=f"svg_{sheet_name}",
            )
        with col_e2:
            st.download_button(
                "Download PDF", pdf_bytes,
                f"{sheet_name}.pdf", "application/pdf",
                key=f"pdf_{sheet_name}",
            )
        with col_e3:
            st.download_button(
                "Download PNG", png_bytes,
                f"{sheet_name}.png", "image/png",
                key=f"png_{sheet_name}",
            )
        for csv_name, csv_bytes_data in stats_csv_data.items():
            with col_e4:
                st.download_button(
                    "Stats CSV", csv_bytes_data,
                    f"{csv_name}.csv", "text/csv",
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
                    fig_b, _, csv_data, _, _ = generate_plot_and_stats(sn)
                    batch_figs[sn] = fig_b
                    batch_csvs.update(csv_data)
                except Exception:
                    pass

            zip_bytes = create_batch_zip(batch_figs, batch_csvs, batch_fmt, export_dpi)
            for f in batch_figs.values():
                plt.close(f)

            st.download_button(
                "Download ZIP", zip_bytes,
                "batch_export.zip", "application/zip",
                key="batch_zip",
            )
