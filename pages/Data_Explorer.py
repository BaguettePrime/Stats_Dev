"""Page 1 – Data Explorer: interactive visualisation and outlier detection."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.shared_ui import (
    render_upload,
    render_condition_setup,
    get_sheet_info,
    get_excluded_subjects,
    render_sidebar_colors,
    render_sidebar_plot_config,
)
from utils.data_loader import get_midpoints, get_valid_subjects, detect_event_gap

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
view_mode = st.sidebar.radio(
    "View mode",
    ["Mean ± SEM", "Individual traces"],
    index=0,
    key="explorer_view_mode",
)
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def _valid_subjects(df: pd.DataFrame) -> list[str]:
    return [s for s in get_valid_subjects(df) if s not in excluded]


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_data
def _mean_sem(values: np.ndarray):
    mean = np.nanmean(values, axis=1)
    n = np.sum(~np.isnan(values), axis=1)
    sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, sem


# ── Plotly builders ──────────────────────────────────────────────────────────

def _build_mean_sem_figure(sheet_name: str, conds: list[str]) -> go.Figure:
    fig = go.Figure()
    plot_type = config["plot_type"]

    for cond in conds:
        df = all_sheets[cond][sheet_name]
        color = colors[cond]
        midpoints = get_midpoints(df)
        valid = _valid_subjects(df)
        if not valid:
            continue
        vals = df[valid].values.astype(float)
        mean, sem = _mean_sem(vals)
        label = display_names[cond]

        if plot_type == "error_bars":
            fig.add_trace(go.Scatter(
                x=midpoints, y=mean,
                error_y=dict(type="data", array=sem, visible=True),
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate="%{x:.1f}s: %{y:.4f} ± %{error_y.array:.4f}<extra>" + label + "</extra>",
            ))
        else:
            # SEM band
            upper = mean + sem
            lower = mean - sem
            fig.add_trace(go.Scatter(
                x=np.concatenate([midpoints, midpoints[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.2),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            # Mean line
            fig.add_trace(go.Scatter(
                x=midpoints, y=mean,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate="%{x:.1f}s: %{y:.4f}<extra>" + label + "</extra>",
            ))

    _apply_layout(fig, sheet_name, conds, hovermode="x unified")
    return fig


def _build_individual_figure(sheet_name: str, conds: list[str]) -> go.Figure:
    fig = go.Figure()

    for cond in conds:
        df = all_sheets[cond][sheet_name]
        color = colors[cond]
        midpoints = get_midpoints(df)
        valid = _valid_subjects(df)
        if not valid:
            continue
        label = display_names[cond]

        for subj in valid:
            vals = df[subj].values.astype(float)
            fig.add_trace(go.Scatter(
                x=midpoints, y=vals,
                mode="lines",
                name=subj,
                legendgroup=cond,
                legendgrouptitle_text=label,
                line=dict(color=color, width=1.2),
                opacity=0.7,
                hovertemplate=(
                    f"<b>{subj}</b><br>"
                    "%{x:.1f}s: %{y:.4f}"
                    f"<extra>{label}</extra>"
                ),
            ))

    _apply_layout(fig, sheet_name, conds, hovermode="closest")
    return fig


def _apply_layout(fig: go.Figure, sheet_name: str, conds: list[str], hovermode: str):
    """Apply shared layout settings to a Plotly figure."""
    ref_df = all_sheets[conds[0]][sheet_name]
    if detect_event_gap(ref_df):
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)

    title = config["custom_title"] if config["custom_title"] else sheet_name
    fig.update_layout(
        title=dict(text=title, font_size=config["font_size_title"]),
        xaxis_title=dict(text=config["x_label"], font_size=config["font_size_axis"]),
        yaxis_title=dict(
            text=config["y_label"] if config["y_label"] else sheet_name,
            font_size=config["font_size_axis"],
        ),
        width=int(config["fig_width"] * 96),
        height=int(config["fig_height"] * 96),
        legend=dict(
            font_size=config["font_size_legend"],
            groupclick="togglegroup",
        ),
        hovermode=hovermode,
        template="plotly_white",
    )
    fig.update_xaxes(tickfont_size=config["font_size_tick"])
    fig.update_yaxes(tickfont_size=config["font_size_tick"])


# ── Visualisation ────────────────────────────────────────────────────────────
st.header("3. Data Visualisation")

for sheet_name in selected_sheets:
    st.subheader(f"Sheet: {sheet_name}")
    conds_with_sheet = [c for c in condition_order if sheet_name in all_sheets[c]]

    if not conds_with_sheet:
        st.warning(f"No conditions contain sheet {sheet_name}.")
        continue

    try:
        # ── Interactive plot ─────────────────────────────────────────────
        if view_mode == "Mean ± SEM":
            fig = _build_mean_sem_figure(sheet_name, conds_with_sheet)
        else:
            fig = _build_individual_figure(sheet_name, conds_with_sheet)

        st.plotly_chart(fig, use_container_width=True)

        # ── Individual data tables ───────────────────────────────────────
        with st.expander(f"Individual data — {sheet_name}", expanded=False):
            for cond in conds_with_sheet:
                df = all_sheets[cond][sheet_name]
                valid = _valid_subjects(df)
                if not valid:
                    continue
                st.write(f"**{display_names[cond]}** ({len(valid)} subjects)")
                show_df = df[valid].copy()
                show_df.index.name = "Time window"
                st.dataframe(show_df, use_container_width=True)

        # ── Subject summary & outlier detection ──────────────────────────
        with st.expander(f"Subject summary — {sheet_name}", expanded=False):
            for cond in conds_with_sheet:
                df = all_sheets[cond][sheet_name]
                valid = _valid_subjects(df)
                if not valid:
                    continue

                values = df[valid].values.astype(float)
                subj_means = np.nanmean(values, axis=0)
                subj_stds = np.nanstd(values, axis=0, ddof=1)
                n_valid_tp = np.sum(~np.isnan(values), axis=0)

                median_val = np.nanmedian(subj_means)
                mad = np.nanmedian(np.abs(subj_means - median_val))
                MAD_SCALE = 1.4826
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

                outliers = summary.loc[
                    summary["Outlier (|Z| > 3.5)"], "Subject"
                ].tolist()
                if outliers:
                    st.warning(
                        f"Potential outliers (MAD-based, |modified Z| > 3.5): "
                        f"{', '.join(outliers)}. "
                        f"Use the sidebar to exclude them."
                    )

        # ── Heatmap ──────────────────────────────────────────────────────
        if show_heatmap:
            with st.expander(f"Subject heatmap — {sheet_name}", expanded=True):
                for cond in conds_with_sheet:
                    df = all_sheets[cond][sheet_name]
                    valid = _valid_subjects(df)
                    if not valid:
                        continue

                    midpoints = get_midpoints(df)
                    data = df[valid].values.astype(float).T

                    fig_h = go.Figure(data=go.Heatmap(
                        z=data,
                        x=[f"{m:.0f}" for m in midpoints],
                        y=valid,
                        colorscale="RdBu_r",
                        hovertemplate=(
                            "Subject: %{y}<br>"
                            "Time: %{x}s<br>"
                            "Value: %{z:.4f}<extra></extra>"
                        ),
                    ))
                    fig_h.update_layout(
                        title=display_names[cond],
                        xaxis_title="Time (s)",
                        height=max(300, len(valid) * 30),
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying {sheet_name}: {e}")
        import traceback
        st.code(traceback.format_exc())
