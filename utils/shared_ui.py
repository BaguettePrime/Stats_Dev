"""Shared UI components used by both pages."""

import streamlit as st
from utils.data_loader import (
    extract_condition_name,
    load_excel_file,
    find_common_sheets,
    find_all_sheets,
    get_valid_subjects,
)
from utils.plotting import DEFAULT_COLORS


def render_upload() -> bool:
    """Render file upload widget and persist data in session state.

    Returns True if data is available.
    """
    uploaded_files = st.file_uploader(
        "Upload Excel files (one per condition)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        file_key = tuple(sorted(f.name for f in uploaded_files))
        if st.session_state.get("_file_key") != file_key:
            all_sheets: dict = {}
            for f in uploaded_files:
                cond = extract_condition_name(f.name)
                all_sheets[cond] = load_excel_file(f)
            st.session_state["all_sheets"] = all_sheets
            st.session_state["sorted_conditions"] = sorted(all_sheets.keys())
            st.session_state["_file_key"] = file_key
            st.session_state["excluded_subjects"] = set()
        return True

    if "all_sheets" in st.session_state:
        return True

    st.info(
        "Upload one or more .xlsx files to get started. "
        "Each file represents one experimental condition."
    )
    return False


def render_condition_setup():
    """Render condition label inputs. Returns (display_names, condition_order)."""
    sorted_conditions = st.session_state["sorted_conditions"]
    cols = st.columns(len(sorted_conditions))
    display_names: dict[str, str] = {}
    order: list[str] = []

    for i, cond in enumerate(sorted_conditions):
        with cols[i]:
            display = st.text_input(
                f"Condition {cond} label",
                value=cond,
                key=f"cond_label_{cond}",
            )
            display_names[cond] = display
            order.append(cond)

    st.session_state["condition_display_names"] = display_names
    st.session_state["condition_order"] = order
    st.caption(
        f"Conditions (in order): {', '.join(display_names[c] for c in order)}"
    )
    return display_names, order


def get_sheet_info():
    """Return (all_sheets, common_sheets, all_sheet_names)."""
    all_sheets = st.session_state["all_sheets"]
    common = find_common_sheets(all_sheets)
    all_names = find_all_sheets(all_sheets)
    return all_sheets, common, all_names


def get_excluded_subjects() -> set:
    """Return the current set of excluded subject names."""
    return st.session_state.get("excluded_subjects", set())


def filter_excluded_from_sheets(all_sheets, excluded):
    """Return a copy of all_sheets with excluded subject columns removed."""
    if not excluded:
        return all_sheets
    filtered: dict = {}
    for cond, sheets in all_sheets.items():
        filtered[cond] = {}
        for name, df in sheets.items():
            keep = [c for c in df.columns if c not in excluded]
            filtered[cond][name] = df[keep]
    return filtered


def render_sidebar_colors(condition_order, display_names):
    """Render colour pickers in the sidebar. Returns {cond: hex_color}."""
    st.sidebar.subheader("Colors")
    colors: dict[str, str] = {}
    for i, cond in enumerate(condition_order):
        default = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        colors[cond] = st.sidebar.color_picker(
            f"{display_names[cond]} color",
            value=default,
            key=f"color_{cond}",
        )
    return colors


def render_sidebar_plot_config():
    """Render plot-configuration widgets in the sidebar. Returns config dict."""
    with st.sidebar.expander("Labels & Fonts", expanded=False):
        custom_title = st.text_input(
            "Title (leave blank for sheet name)", value="", key="custom_title"
        )
        y_label = st.text_input(
            "Y-axis label (leave blank for sheet name)", value="", key="y_label"
        )
        x_label = st.text_input("X-axis label", value="Time (s)", key="x_label")
        fs_title = st.slider("Title font size", 8, 28, 14, key="fs_title")
        fs_axis = st.slider("Axis label font size", 6, 24, 12, key="fs_axis")
        fs_tick = st.slider("Tick label font size", 6, 20, 10, key="fs_tick")
        fs_legend = st.slider("Legend font size", 6, 20, 10, key="fs_legend")

    with st.sidebar.expander("Figure Size", expanded=False):
        fig_w = st.slider("Width (inches)", 4.0, 20.0, 10.0, 0.5, key="fig_w")
        fig_h = st.slider("Height (inches)", 3.0, 15.0, 6.0, 0.5, key="fig_h")

    legend_pos = st.sidebar.selectbox(
        "Legend position",
        ["upper right", "upper left", "lower right", "lower left", "outside right"],
        index=0,
    )

    plot_type = st.sidebar.radio(
        "Plot type",
        ["Line + shaded SEM", "Line + error bars"],
        index=0,
    )

    return {
        "custom_title": custom_title,
        "y_label": y_label,
        "x_label": x_label,
        "font_size_title": fs_title,
        "font_size_axis": fs_axis,
        "font_size_tick": fs_tick,
        "font_size_legend": fs_legend,
        "fig_width": fig_w,
        "fig_height": fig_h,
        "legend_position": legend_pos,
        "plot_type": "error_bars" if plot_type == "Line + error bars" else "shaded_sem",
    }
