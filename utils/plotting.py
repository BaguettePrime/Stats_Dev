"""Figure generation with matplotlib."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.data_loader import get_midpoints, detect_event_gap

# Default colorblind-friendly palette (tab10 subset)
DEFAULT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

SIGNIFICANCE_COLORS = ["#888888", "#555555", "#222222"]


def create_figure(
    condition_data: dict[str, dict],
    sheet_name: str,
    config: dict,
    stats_results: dict | None = None,
) -> plt.Figure:
    """Create a publication-ready time series figure.

    Args:
        condition_data: {condition_name: {'df': DataFrame, 'color': str}}
        sheet_name: name of the sheet being plotted
        config: plot configuration dict with keys:
            - plot_type: 'shaded_sem' or 'error_bars'
            - y_label, x_label, title
            - font_size_title, font_size_axis, font_size_tick, font_size_legend
            - fig_width, fig_height
            - legend_position
            - show_event_line
            - show_p_curve
            - sig_thresholds: list of (threshold, label) e.g. [(0.05, '*'), (0.01, '**')]
            - comparison_labels: list of str for each stats result
        stats_results: dict mapping comparison label to stats dict from run_tests_across_timepoints
            Each stats dict also needs 'midpoints' key with the x-values for significant markers
    """
    show_p = config.get("show_p_curve", False) and stats_results
    if show_p:
        fig, (ax, ax_p) = plt.subplots(
            2, 1,
            figsize=(config["fig_width"], config["fig_height"]),
            height_ratios=[3, 1],
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.08)
    else:
        fig, ax = plt.subplots(figsize=(config["fig_width"], config["fig_height"]))
        ax_p = None

    # Plot each condition
    for cond_name, cond_info in condition_data.items():
        df = cond_info["df"]
        color = cond_info["color"]
        midpoints = get_midpoints(df)
        valid_subjects = [c for c in df.columns if not df[c].isna().all()]
        values = df[valid_subjects].values.astype(float)
        mean = np.nanmean(values, axis=1)
        sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(
            np.sum(~np.isnan(values), axis=1)
        )

        if config.get("plot_type") == "error_bars":
            ax.errorbar(
                midpoints, mean, yerr=sem,
                label=cond_name, color=color,
                linewidth=1.5, capsize=3, capthick=1,
                marker="o", markersize=3,
            )
        else:  # shaded SEM (default)
            ax.plot(midpoints, mean, label=cond_name, color=color, linewidth=1.5)
            ax.fill_between(
                midpoints, mean - sem, mean + sem,
                color=color, alpha=0.2,
            )

    # Event line at x=0
    any_df = next(iter(condition_data.values()))["df"]
    if config.get("show_event_line", True) and detect_event_gap(any_df):
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        if ax_p is not None:
            ax_p.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    # Significance bars
    if stats_results:
        sig_thresholds = config.get("sig_thresholds", [(0.05, "p < 0.05")])
        y_min, y_max = ax.get_ylim()
        bar_height = (y_max - y_min) * 0.02
        legend_extras = []

        for comp_idx, (comp_label, sr) in enumerate(stats_results.items()):
            midpts = sr["midpoints"]
            p_corr = sr["p_corrected"]

            for thresh_idx, (threshold, thresh_label) in enumerate(sig_thresholds):
                sig_mask = p_corr < threshold
                if not np.any(sig_mask):
                    continue

                bar_y = y_max + bar_height * (comp_idx * len(sig_thresholds) + thresh_idx + 0.5)
                color = SIGNIFICANCE_COLORS[min(thresh_idx, len(SIGNIFICANCE_COLORS) - 1)]

                # Draw horizontal bars at significant timepoints
                sig_indices = np.where(sig_mask)[0]
                if len(sig_indices) == 0:
                    continue

                # Group consecutive significant timepoints into spans
                spans = _group_consecutive(sig_indices, midpts)
                for x_start, x_end in spans:
                    ax.fill_between(
                        [x_start, x_end],
                        [bar_y - bar_height / 2] * 2,
                        [bar_y + bar_height / 2] * 2,
                        color=color, alpha=0.7,
                    )

                display_label = f"{comp_label} {thresh_label}" if comp_label else thresh_label
                legend_extras.append(
                    Line2D([0], [0], color=color, linewidth=4, alpha=0.7, label=display_label)
                )

        # Extend y-axis to show sig bars
        if legend_extras:
            new_ymax = y_max + bar_height * (
                len(stats_results) * len(sig_thresholds) + 1
            )
            ax.set_ylim(y_min, new_ymax)

        # P-value curve subplot
        if ax_p is not None:
            for comp_label, sr in stats_results.items():
                midpts = sr["midpoints"]
                p_corr = sr["p_corrected"]
                ax_p.plot(midpts, p_corr, linewidth=1, label=comp_label or "p-value")
            for threshold, thresh_label in sig_thresholds:
                ax_p.axhline(
                    y=threshold, color="red", linestyle="--",
                    linewidth=0.8, alpha=0.6, label=f"\u03b1 = {threshold}",
                )
            ax_p.set_ylabel("p-value (corrected)", fontsize=config["font_size_axis"])
            ax_p.set_yscale("log")
            ax_p.tick_params(labelsize=config["font_size_tick"])
            ax_p.legend(fontsize=max(config["font_size_legend"] - 2, 6))

    # Labels and styling
    ax.set_title(config.get("title", sheet_name), fontsize=config["font_size_title"])
    target_ax = ax_p if ax_p is not None else ax
    target_ax.set_xlabel(config.get("x_label", "Time (s)"), fontsize=config["font_size_axis"])
    ax.set_ylabel(config.get("y_label", sheet_name), fontsize=config["font_size_axis"])
    ax.tick_params(labelsize=config["font_size_tick"])

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if stats_results:
        sig_thresholds = config.get("sig_thresholds", [(0.05, "p < 0.05")])
        for comp_label, sr in stats_results.items():
            for thresh_idx, (threshold, thresh_label) in enumerate(sig_thresholds):
                if np.any(sr["p_corrected"] < threshold):
                    color = SIGNIFICANCE_COLORS[min(thresh_idx, len(SIGNIFICANCE_COLORS) - 1)]
                    display_label = f"{comp_label} {thresh_label}" if comp_label else thresh_label
                    handles.append(
                        Line2D([0], [0], color=color, linewidth=4, alpha=0.7)
                    )
                    labels.append(display_label)

    legend_pos = config.get("legend_position", "upper right")
    if legend_pos == "outside right":
        ax.legend(handles, labels, fontsize=config["font_size_legend"],
                  bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        loc_map = {
            "upper right": "upper right",
            "upper left": "upper left",
            "lower right": "lower right",
            "lower left": "lower left",
        }
        ax.legend(handles, labels, fontsize=config["font_size_legend"],
                  loc=loc_map.get(legend_pos, "upper right"))

    fig.tight_layout()
    return fig


def _group_consecutive(indices: np.ndarray, midpoints: np.ndarray) -> list[tuple[float, float]]:
    """Group consecutive indices into (x_start, x_end) spans using midpoints."""
    if len(indices) == 0:
        return []

    spans = []
    start_idx = indices[0]
    prev_idx = indices[0]

    # Compute half-widths for extending bars
    all_mids = sorted(set(midpoints[~np.isnan(midpoints)]))

    for i in range(1, len(indices)):
        if indices[i] == prev_idx + 1:
            prev_idx = indices[i]
        else:
            spans.append(_make_span(start_idx, prev_idx, midpoints, all_mids))
            start_idx = indices[i]
            prev_idx = indices[i]
    spans.append(_make_span(start_idx, prev_idx, midpoints, all_mids))
    return spans


def _make_span(start_idx, end_idx, midpoints, all_mids):
    """Create a span from start to end index, extending by half a bin width."""
    x_start = midpoints[start_idx]
    x_end = midpoints[end_idx]
    # Extend by half the local bin width
    if len(all_mids) > 1:
        idx_s = all_mids.index(x_start) if x_start in all_mids else 0
        idx_e = all_mids.index(x_end) if x_end in all_mids else len(all_mids) - 1
        if idx_s > 0:
            hw_start = (all_mids[idx_s] - all_mids[idx_s - 1]) / 2
        else:
            hw_start = (all_mids[1] - all_mids[0]) / 2 if len(all_mids) > 1 else 0
        if idx_e < len(all_mids) - 1:
            hw_end = (all_mids[idx_e + 1] - all_mids[idx_e]) / 2
        else:
            hw_end = (all_mids[-1] - all_mids[-2]) / 2 if len(all_mids) > 1 else 0
        x_start -= hw_start
        x_end += hw_end
    return (x_start, x_end)
