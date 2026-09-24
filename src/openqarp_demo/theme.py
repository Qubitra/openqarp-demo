"""The dashboard's light corporate theme: a Plotly template plus the app stylesheet.

A navy and slate palette with one accent, reserved for the chosen basket. The five
sector hues are a categorical set checked for colour-vision-deficiency separation; they
appear only on the market chart, never beside the accent.
"""

from __future__ import annotations

from importlib.resources import files

import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE_NAME = "qubitra_light"
FONT_FAMILY = "Inter, 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"

NAVY = "#14284B"
INK = "#1E293B"
INK_MUTED = "#5B6B82"
GRID = "#E6EAF0"
SURFACE = "#FFFFFF"
PAGE = "#F4F6F9"

#: The chosen basket. Nothing else on screen wears it.
ACCENT = "#E8590C"
#: The complement side of a partition, deliberately neutral.
COMPLEMENT = "#5B6B82"
#: A platform series and its local counterpart.
PLATFORM = "#2B5BB5"
LOCAL = "#0B9A9E"
REFERENCE = "#94A3B8"

SECTOR_COLORS = {
    "TECH": "#2B5BB5",
    "ENRG": "#0B9A9E",
    "FINL": "#7B61D9",
    "GOLD": "#C2841A",
    "HLTH": "#D0508A",
}


def plotly_template() -> go.layout.Template:
    """The Plotly template every chart in the app is drawn with."""
    axis = {
        "gridcolor": GRID,
        "linecolor": GRID,
        "zerolinecolor": GRID,
        "tickfont": {"color": INK_MUTED, "size": 12},
        "title": {"font": {"color": INK_MUTED, "size": 12}},
        "ticks": "",
    }
    return go.layout.Template(
        layout=go.Layout(
            font={"family": FONT_FAMILY, "color": INK, "size": 13},
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=[PLATFORM, LOCAL, ACCENT, COMPLEMENT, REFERENCE],
            xaxis=axis,
            yaxis=axis,
            margin={"l": 48, "r": 16, "t": 16, "b": 40},
            hoverlabel={
                "bgcolor": SURFACE,
                "bordercolor": GRID,
                "font": {"family": FONT_FAMILY, "color": INK, "size": 12},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0.0,
                "font": {"color": INK_MUTED, "size": 12},
                "bgcolor": "rgba(0,0,0,0)",
            },
        )
    )


def register_template() -> str:
    """Register the template with Plotly and make it the default; returns its name."""
    pio.templates[TEMPLATE_NAME] = plotly_template()
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def stylesheet() -> str:
    """The app's custom CSS, as shipped next to the notebook."""
    return files("openqarp_demo").joinpath("theme.css").read_text(encoding="utf-8")
