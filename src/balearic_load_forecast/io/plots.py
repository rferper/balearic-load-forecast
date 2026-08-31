"""Rendering forecasts to a PNG.

Both series are MW and share one y-axis. The two colours are an adjacent pair
from the categorical palette, checked for colour-vision-deficient separation.
"""

from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from ..domain.features import LOCAL_TZ

# No display on a server or in CI.
matplotlib.use("Agg")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SERIES_ACTUAL = "#2a78d6"  # categorical slot 1, blue
SERIES_FORECAST = "#eb6834"  # categorical slot 2, orange

LINE_WIDTH = 2.0
FIGURE_SIZE = (11.0, 5.0)


def _style_axes(ax: plt.Axes) -> None:
    """Apply the recessive grid and axis styling."""
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    day: date,
    model_id: str,
    path: Path,
    dpi: int = 150,
) -> Path:
    """Draw the forecast against recent actual demand and save it."""
    local_actual = actual.tz_convert(LOCAL_TZ)
    local_forecast = forecast.tz_convert(LOCAL_TZ)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=dpi, facecolor=SURFACE)
    _style_axes(ax)

    ax.plot(
        local_actual.index,
        local_actual.to_numpy(),
        color=SERIES_ACTUAL,
        linewidth=LINE_WIDTH,
        label="Actual demand",
    )
    ax.plot(
        local_forecast.index,
        local_forecast.to_numpy(),
        color=SERIES_FORECAST,
        linewidth=LINE_WIDTH,
        label=f"Forecast for {day}",
    )

    # Headroom so the legend does not sit across the marks.
    lowest = min(local_actual.min(), local_forecast.min())
    highest = max(local_actual.max(), local_forecast.max())
    span = highest - lowest
    ax.set_ylim(lowest - 0.05 * span, highest + 0.22 * span)

    # The peak is the number worth labelling. It sits at the right edge, so
    # the label goes to its left to stay inside the frame.
    peak_at = local_forecast.idxmax()
    peak_mw = float(local_forecast.max())
    ax.annotate(
        f"peak {peak_mw:,.0f} MW",
        xy=(peak_at, peak_mw),
        xytext=(-8, 6),
        textcoords="offset points",
        horizontalalignment="right",
        color=TEXT_PRIMARY,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(
        f"Balearic day-ahead load forecast - {day}",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    ax.text(
        0.0,
        1.02,
        f"Hourly demand, local time ({LOCAL_TZ}) - model {model_id}",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.set_ylabel("MW", color=TEXT_SECONDARY, fontsize=9)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b", tz=local_actual.index.tz)
    )

    legend = ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    logger.info("wrote figure {}", path)
    return path
