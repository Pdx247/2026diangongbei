"""Shared academic plotting style for the B problem figures."""

from __future__ import annotations

FONT_FAMILIES = [
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "STHeiti",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]

COLORS = {
    "ink": "#1F2933",
    "muted": "#5B677A",
    "axis": "#A8B1BD",
    "grid": "#E6EAF0",
    "panel": "#F7F9FB",
    "blue": "#2F5D8C",
    "cyan": "#4A90A4",
    "green": "#6BA368",
    "gold": "#C69C3D",
    "orange": "#C9713C",
    "red": "#B85C5C",
    "purple": "#7664A8",
    "gray": "#C7CED8",
    "light_gray": "#E8EDF3",
}

SERIES_PALETTE = [
    COLORS["blue"],
    COLORS["cyan"],
    COLORS["green"],
    COLORS["gold"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["red"],
]

TYPE_PALETTE = {
    "自理": COLORS["blue"],
    "半失能": COLORS["gold"],
    "失能": COLORS["red"],
}

SERVICE_PALETTE = {
    "助餐": COLORS["blue"],
    "日间照料": COLORS["cyan"],
    "上门护理": COLORS["green"],
    "康复理疗": COLORS["gold"],
    "助浴": COLORS["orange"],
    "紧急救助": COLORS["purple"],
}

SCALE_PALETTE = {
    "小型": COLORS["green"],
    "中型": COLORS["gold"],
    "大型": COLORS["red"],
}

FIGSIZE_WIDE = (8.6, 4.9)
FIGSIZE_TALL = (8.6, 5.8)
FIGSIZE_COMPACT = (7.4, 4.9)
FIGSIZE_NETWORK = (7.6, 6.0)
FIGSIZE_HEATMAP = (8.0, 4.9)


def configure_matplotlib(plt, fm=None) -> None:
    """Apply the common visual system to matplotlib."""
    chosen_font = None
    if fm is not None:
        for font in FONT_FAMILIES:
            try:
                fm.findfont(font, fallback_to_default=False)
                chosen_font = font
                break
            except ValueError:
                continue

    sans_serif = [chosen_font] + FONT_FAMILIES if chosen_font else FONT_FAMILIES
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans_serif,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.0,
            "figure.constrained_layout.use": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_axis_style(ax, grid: str | None = "y") -> None:
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3.0, width=0.8, color=COLORS["axis"])


def apply_twin_axis_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color(COLORS["axis"])
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=3.0, width=0.8, color=COLORS["axis"])


def save_figure(fig, path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.04)


def make_colormap(name: str, colors: list[str]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, colors)


def annotate_vertical_bars(ax, bars, fmt: str = "{:.2f}", padding: float = 3.0, fontsize: float = 8.5) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLORS["muted"],
        )


def annotate_horizontal_bars(ax, bars, fmt: str = "{:.2f}", padding: float = 4.0, fontsize: float = 8.5) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            fmt.format(width),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(padding, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=COLORS["muted"],
        )
