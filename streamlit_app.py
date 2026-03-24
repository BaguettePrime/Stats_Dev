"""Page 1 -- Data Explorer: visualisation, individual data, outlier detection."""

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
    render_sidebar_colors,
    render_sidebar_plot_config,
)
from utils.data_loader import get_midpoints, get_valid_subjects, detect_event_gap
from utils.export import fig_to_svg, fig_to_pdf, fig_to_png

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

# ── Upload & Setup ───────────────────────────────────────────────────────────
st.header("1. Upload Data")
if not render_upload():
    st.stop()

st.header("2. Condition Setup")
display_names, condition_order = render_condition_setup()

all_sheets, common_sheets, all_sheet_names = get_sheet_info()
if not all_sheet_names:
    st.error("No sheets found in uploaded files.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Explorer Configuration")

selected_sheets = st.sidebar.multiselect(
    "Sheets to display",
    options=all_sheet_names,
    default=common_sheets[:1] if common_sheets else all_sheet_names[:1],
    key="explorer_sheets",
)
if not selected_sheets:
    st.warning("Select at least one sheet.")
    st.stop()

colors = render_sidebar_colors(condition_order, display_names)
config = render_sidebar_plot_config()

st.sidebar.markdown("---")
show_individual = st.sidebar.checkbox("Show individual traces", value=True)
show_heatmap = st.sidebar.checkbox("Show subject heatmap", value=False)

# ── Subject Exclusion ────────────────────────────────────────────────────────
st.sidebar.header("Subject Exclusion")
all_subjects: set[str] = set()
for cond in condition_order:
    for sn in all_sheet_names:
        if sn in all_sheets[cond]:
            all_subjects.update(get_valid_subjects(all_sheets[cond][sn]))

current_excluded = get_excluded_subjects()
excluded_selection = st.sidebar.multiselect(
    "Exclude subjects",
    sorted(all_subjects),
    default=sorted(current_excluded & all_subjects),
    key="exclude_subjects_selector",
    help="Excluded subjects are also removed from the Statistical Analysis page.",
)
st.session_state["excluded_subjects"] = set(excluded_selection)
excluded = set(excluded_selection)

if excluded:
    st.sidebar.caption(f"Excluding: {', '.join(sorted(excluded))}")

# ── Export settings ──────────────────────────────────────────────────────────
st.sidebar.header("Export")
export_dpi = st.sidebar.number_input("PNG DPI", 72, 600, 300, 50, key="explorer_dpi")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _valid_subjects(df):
    """Valid subjects minus excluded ones."""
    return [s for s in get_valid_subjects(df) if s not in excluded]


def _mean_sem(df, valid):
    """Compute mean and SEM from valid columns."""
    vals = df[valid].values.astype(float)
    mean = np.nanmean(vals, axis=1)
    n = np.sum(~np.isnan(vals), axis=1)
    sem = np.nanstd(vals, axis=1, ddof=1) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, sem


def _build_explorer_figure(sheet_name, conds):
    """Build the explorer figure with mean +/- SEM and optional individual traces."""
    fig, ax = plt.subplots(figsize=(config["fig_width"], config["fig_height"]))

    for cond in conds:
        df = all_sheets[cond][sheet_name]
        color = colors[cond]
        midpoints = get_midpoints(df)
        valid = _valid_subjects(df)
        if not valid:
            continue
        mean, sem = _mean_sem(df, valid)

        # Individual traces
        if show_individual:
            for subj in valid:
                ax.plot(
                    midpoints, df[subj].values.astype(float),
                    color=color, alpha=0.15, linewidth=0.7,
                )

        # Mean +/- SEM
        label = display_names[cond]
        if config["plot_type"] == "error_bars":
            ax.errorbar(
                midpoints, mean, yerr=sem, label=label, color=color,
                linewidth=2, capsize=3, capthick=1, marker="o", markersize=3,
            )
        else:
            ax.plot(midpoints, mean, label=label, color=color, linewidth=2)
            ax.fill_between(midpoints, mean - sem, mean + sem, color=color, alpha=0.2)

    # Event line at t = 0
    ref_df = all_sheets[conds[0]][sheet_name]
    if detect_event_gap(ref_df):
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    title = config["custom_title"] if config["custom_title"] else sheet_name
    ax.set_title(title, fontsize=config["font_size_title"])
    ax.set_ylabel(
        config["y_label"] if config["y_label"] else sheet_name,
        fontsize=config["font_size_axis"],
    )
    ax.set_xlabel(config["x_label"], fontsize=config["font_size_axis"])
    ax.tick_params(labelsize=config["font_size_tick"])

    lp = config["legend_position"]
    if lp == "outside right":
        ax.legend(
            fontsize=config["font_size_legend"],
            bbox_to_anchor=(1.02, 1), loc="upper left",
        )
    else:
        ax.legend(fontsize=config["font_size_legend"], loc=lp)

    fig.tight_layout()
    return fig


# ── Visualisation ────────────────────────────────────────────────────────────
st.header("3. Data Visualisation")

for sheet_name in selected_sheets:
    st.subheader(f"Sheet: {sheet_name}")
    conds_with_sheet = [c for c in condition_order if sheet_name in all_sheets[c]]

    if not conds_with_sheet:
        st.warning(f"No conditions contain sheet {sheet_name}.")
        continue

    try:
        fig = _build_explorer_figure(sheet_name, conds_with_sheet)

        # Cache export bytes before closing
        svg_bytes = fig_to_svg(fig)
        pdf_bytes = fig_to_pdf(fig)
        png_bytes = fig_to_png(fig, export_dpi)

        st.pyplot(fig)
        plt.close(fig)

        # Export buttons
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.download_button(
                "Download SVG", svg_bytes,
                f"{sheet_name}.svg", "image/svg+xml",
                key=f"exp_svg_{sheet_name}",
            )
        with col_e2:
            st.download_button(
                "Download PDF", pdf_bytes,
                f"{sheet_name}.pdf", "application/pdf",
                key=f"exp_pdf_{sheet_name}",
            )
        with col_e3:
            st.download_button(
                "Download PNG", png_bytes,
                f"{sheet_name}.png", "image/png",
                key=f"exp_png_{sheet_name}",
            )

        # ── Individual Data Tables ───────────────────────────────────────
        with st.expander(f"Individual data \u2014 {sheet_name}", expanded=False):
            for cond in conds_with_sheet:
                df = all_sheets[cond][sheet_name]
                valid = _valid_subjects(df)
                if not valid:
                    continue
                st.write(f"**{display_names[cond]}** ({len(valid)} subjects)")
                show_df = df[valid].copy()
                show_df.index.name = "Time window"
                st.dataframe(show_df, use_container_width=True)

        # ── Subject Summary & Outlier Detection ──────────────────────────
        with st.expander(f"Subject summary \u2014 {sheet_name}", expanded=False):
            for cond in conds_with_sheet:
                df = all_sheets[cond][sheet_name]
                valid = _valid_subjects(df)
                if not valid:
                    continue

                values = df[valid].values.astype(float)
                subj_means = np.nanmean(values, axis=0)
                subj_stds = np.nanstd(values, axis=0, ddof=1)
                n_valid_tp = np.sum(~np.isnan(values), axis=0)

                # MAD-based outlier detection (modified Z-score)
                median_val = np.nanmedian(subj_means)
                mad = np.nanmedian(np.abs(subj_means - median_val))
                MAD_SCALE = 1.4826  # consistency constant for normality
                if mad > 0:
                    mod_z = np.abs((subj_means - median_val) / (mad * MAD_SCALE))
                else:
                    mod_z = np.zeros_like(subj_means)

                summary = pd.DataFrame({
                    "Subject": valid,
                    "Mean": np.round(subj_means, 4),
                    "Std": np.round(subj_stds, 4),
                    "N valid timepoints": n_valid_tp.astype(int),
                    "Modified Z-score": np.round(mod_z, 2),
                    "Outlier (|Z| > 3.5)": mod_z > 3.5,
                })

                st.write(f"**{display_names[cond]}**")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                outliers = summary.loc[summary["Outlier (|Z| > 3.5)"], "Subject"].tolist()
                if outliers:
                    st.warning(
                        f"Potential outliers (MAD-based, |modified Z| > 3.5): "
                        f"{', '.join(outliers)}. "
                        f"Use the sidebar to exclude them."
                    )

        # ── Heatmap ──────────────────────────────────────────────────────
        if show_heatmap:
            with st.expander(f"Subject heatmap \u2014 {sheet_name}", expanded=True):
                for cond in conds_with_sheet:
                    df = all_sheets[cond][sheet_name]
                    valid = _valid_subjects(df)
                    if not valid:
                        continue

                    midpoints = get_midpoints(df)
                    fig_h, ax_h = plt.subplots(
                        figsize=(config["fig_width"], max(2.0, len(valid) * 0.4))
                    )
                    data = df[valid].values.astype(float).T
                    im = ax_h.imshow(
                        data, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                    )
                    ax_h.set_yticks(range(len(valid)))
                    ax_h.set_yticklabels(valid, fontsize=8)

                    n_ticks = min(15, len(midpoints))
                    tick_idx = np.linspace(0, len(midpoints) - 1, n_ticks, dtype=int)
                    ax_h.set_xticks(tick_idx)
                    ax_h.set_xticklabels(
                        [f"{midpoints[i]:.0f}" for i in tick_idx], fontsize=8,
                    )
                    ax_h.set_xlabel("Time (s)")
                    ax_h.set_title(
                        display_names[cond], fontsize=config["font_size_title"],
                    )
                    plt.colorbar(im, ax=ax_h, shrink=0.8)
                    fig_h.tight_layout()
                    st.pyplot(fig_h)
                    plt.close(fig_h)

    except Exception as e:
        st.error(f"Error displaying {sheet_name}: {e}")
        import traceback
        st.code(traceback.format_exc())
