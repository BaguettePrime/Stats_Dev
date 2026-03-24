"""SVG/PDF/PNG/CSV export utilities."""

import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fig_to_svg(fig: plt.Figure) -> bytes:
    """Export figure to SVG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def fig_to_pdf(fig: plt.Figure) -> bytes:
    """Export figure to PDF bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def fig_to_png(fig: plt.Figure, dpi: int = 300) -> bytes:
    """Export figure to PNG bytes at specified DPI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def stats_to_csv(
    timepoints: list[str],
    midpoints: np.ndarray,
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
    condition_a_name: str,
    condition_b_name: str,
    stats_result: dict | None = None,
) -> bytes:
    """Export statistics table as CSV bytes."""
    data = {
        "timepoint": timepoints,
        "midpoint": midpoints,
        f"{condition_a_name}_mean": mean_a,
        f"{condition_a_name}_sem": sem_a,
        f"{condition_b_name}_mean": mean_b,
        f"{condition_b_name}_sem": sem_b,
    }
    if stats_result is not None:
        data["test_statistic"] = stats_result["test_stats"]
        data["p_value"] = stats_result["p_values"]
        data["p_value_corrected"] = stats_result["p_corrected"]
        data["significant"] = stats_result["significant"]

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def create_batch_zip(
    figures: dict[str, plt.Figure],
    stats_csvs: dict[str, bytes],
    fmt: str = "svg",
    dpi: int = 300,
) -> bytes:
    """Create a ZIP file containing all figures and stats tables.

    Args:
        figures: {sheet_name: matplotlib Figure}
        stats_csvs: {sheet_name: CSV bytes}
        fmt: image format ('svg', 'pdf', 'png')
        dpi: DPI for PNG export
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figures.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            if fmt == "svg":
                zf.writestr(f"{safe_name}.svg", fig_to_svg(fig))
            elif fmt == "pdf":
                zf.writestr(f"{safe_name}.pdf", fig_to_pdf(fig))
            else:
                zf.writestr(f"{safe_name}.png", fig_to_png(fig, dpi))

        for name, csv_data in stats_csvs.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            zf.writestr(f"{safe_name}_stats.csv", csv_data)

    buf.seek(0)
    return buf.getvalue()
