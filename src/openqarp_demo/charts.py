"""Plotly figures for the dashboard, all drawn with the ``qubitra_light`` template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from openqarp_demo.live import JobPoint
from openqarp_demo.market import SECTORS, Market
from openqarp_demo.theme import (
    ACCENT,
    COMPLEMENT,
    GRID,
    INK_MUTED,
    LOCAL,
    NAVY,
    PLATFORM,
    REFERENCE,
    SECTOR_COLORS,
    TEMPLATE_NAME,
    register_template,
)

register_template()

Positions = Mapping[int, tuple[float, float]]


def market_layout(market: Market) -> Positions:
    """Sectors as clusters on a ring, each sector's assets on a small circle of its own.

    Same-sector assets are near-cliques, so a force layout folds each sector into a point;
    placing the clusters explicitly keeps every asset and every cross-sector edge visible.
    """
    sectors = list(dict.fromkeys(market.sectors))
    pos: dict[int, tuple[float, float]] = {}
    for s_index, sector in enumerate(sectors):
        angle = np.pi / 2 - 2 * np.pi * s_index / len(sectors)
        cx, cy = 1.0 * np.cos(angle), 1.0 * np.sin(angle)
        members = [i for i, name in enumerate(market.sectors) if name == sector]
        radius = 0.12 + 0.03 * len(members)
        for m_index, node in enumerate(members):
            theta = angle + np.pi + 2 * np.pi * m_index / len(members)
            pos[node] = (float(cx + radius * np.cos(theta)), float(cy + radius * np.sin(theta)))
    return pos


def sector_ring_layout(market: Market) -> Positions:
    """Assets on a circle in ticker order, so same-sector pairs sit side by side."""
    n = market.n_assets
    return {
        i: (
            float(np.cos(np.pi / 2 - 2 * np.pi * i / n)),
            float(np.sin(np.pi / 2 - 2 * np.pi * i / n)),
        )
        for i in range(n)
    }


def _edge_trace(
    market: Market, pos: Positions, edges: Sequence[tuple[int, int]], color: str, width: float
) -> go.Scatter:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for u, v in edges:
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def _frame(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=height,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        hovermode="closest",
    )
    return fig


def market_network(market: Market, highlight: Sequence[str] = ()) -> go.Figure:
    """The correlation network: sectors as colours, the QAOA sub-market ringed."""
    pos = market_layout(market)
    fig = go.Figure()
    fig.add_trace(_edge_trace(market, pos, list(market.graph.edges()), GRID, 0.8))
    degree = dict(market.graph.degree())
    for sector in SECTORS:
        nodes = [i for i, name in enumerate(market.sectors) if name == sector]
        fig.add_trace(
            go.Scatter(
                x=[pos[i][0] for i in nodes],
                y=[pos[i][1] for i in nodes],
                mode="markers",
                name=sector,
                marker={
                    "size": 13,
                    "color": SECTOR_COLORS[sector],
                    "line": {
                        "color": [
                            NAVY if market.tickers[i] in highlight else "#FFFFFF" for i in nodes
                        ],
                        "width": [3 if market.tickers[i] in highlight else 1.5 for i in nodes],
                    },
                },
                text=[market.tickers[i] for i in nodes],
                customdata=[degree[i] for i in nodes],
                hovertemplate="<b>%{text}</b><br>" + sector + "<br>%{customdata} correlated peers"
                "<extra></extra>",
            )
        )
    if highlight:
        # A legend key for the ring; the names are on hover, and on the partition chart.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="QAOA market (ringed)",
                marker={"size": 13, "color": "#FFFFFF", "line": {"color": NAVY, "width": 3}},
            )
        )
    return _frame(fig, 460)


def partition_graph(
    market: Market, basket: Sequence[int], pos: Positions | None = None, labels: bool = True
) -> go.Figure:
    """A partition drawn on the market graph, the diversified basket in the accent."""
    pos = pos or (sector_ring_layout(market) if market.n_assets <= 12 else market_layout(market))
    inside = set(basket)
    fig = go.Figure()
    same_side = [(u, v) for u, v in market.graph.edges() if (u in inside) == (v in inside)]
    cut = [(u, v) for u, v in market.graph.edges() if (u in inside) != (v in inside)]
    strong_cut = [(u, v) for u, v in cut if abs(market.graph[u][v]["weight"]) >= 0.3]
    fig.add_trace(_edge_trace(market, pos, same_side, GRID, 1.0))
    fig.add_trace(_edge_trace(market, pos, strong_cut, REFERENCE, 2.0))
    for name, members, color in (
        ("Diversified basket", sorted(inside), ACCENT),
        ("Complement", [i for i in range(market.n_assets) if i not in inside], COMPLEMENT),
    ):
        fig.add_trace(
            go.Scatter(
                x=[pos[i][0] for i in members],
                y=[pos[i][1] for i in members],
                mode="markers+text" if labels else "markers",
                name=name,
                marker={
                    "size": 18 if labels else 11,
                    "color": color,
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                text=[market.tickers[i] for i in members],
                textposition="top center",
                textfont={"size": 11, "color": INK_MUTED},
                hovertemplate="<b>%{text}</b><br>" + name + "<extra></extra>",
            )
        )
    return _frame(fig, 380 if labels else 440)


def convergence(
    points: Sequence[JobPoint],
    *,
    reference: float | None,
    reference_label: str,
    expected_jobs: int | None = None,
    y_title: str = "Expected cut",
) -> go.Figure:
    """The objective as each job completes, with the best so far and a reference line."""
    evaluated = [(p.job, p.value) for p in points if p.value is not None]
    fig = go.Figure()
    if evaluated:
        jobs, values = zip(*evaluated, strict=True)
        best = list(np.maximum.accumulate(values))
        fig.add_trace(
            go.Scatter(
                x=jobs,
                y=values,
                mode="lines+markers",
                name="Objective per job",
                line={"color": PLATFORM, "width": 2},
                marker={"size": 8, "color": PLATFORM, "line": {"color": "#FFFFFF", "width": 2}},
                hovertemplate="Job %{x}<br>%{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=jobs,
                y=best,
                mode="lines",
                name="Best so far",
                line={"color": NAVY, "width": 2, "shape": "hv", "dash": "dot"},
                hovertemplate="Best after job %{x}<br>%{y:.4f}<extra></extra>",
            )
        )
    if reference is not None:
        fig.add_hline(
            y=reference,
            line={"color": REFERENCE, "width": 1.5, "dash": "dash"},
            annotation_text=f"{reference_label} {reference:.3f}",
            annotation_position="bottom right",
            annotation_font={"color": INK_MUTED, "size": 11},
        )
    last_job = max((p.job for p in points), default=0)
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=340,
        xaxis={
            "title": {"text": "Platform job"},
            "range": [0.5, max(last_job, expected_jobs or 1) + 0.5],
        },
        yaxis={"title": {"text": y_title}},
        hovermode="x unified",
        margin={"l": 56, "r": 16, "t": 36, "b": 44},
    )
    return fig


def side_by_side(
    local: Sequence[JobPoint], platform: Sequence[JobPoint], optimum: float
) -> go.Figure:
    """The same QAOA on a laptop and on the platform, job for job."""
    fig = go.Figure()
    for name, points, color, dash, size in (
        ("Local QarpEngine", local, LOCAL, "dash", 11),
        ("Qubitra platform", platform, PLATFORM, "solid", 7),
    ):
        evaluated = [(p.job, p.value) for p in points if p.value is not None]
        if not evaluated:
            continue
        jobs, values = zip(*evaluated, strict=True)
        fig.add_trace(
            go.Scatter(
                x=jobs,
                y=values,
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": size, "color": color, "line": {"color": "#FFFFFF", "width": 2}},
                hovertemplate=name + "<br>Job %{x}: %{y:.6f}<extra></extra>",
            )
        )
    fig.add_hline(y=optimum, line={"color": REFERENCE, "width": 1.5, "dash": "dash"})
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=300,
        xaxis={"title": {"text": "Job"}},
        yaxis={"title": {"text": "Expected cut"}},
        hovermode="x unified",
        margin={"l": 56, "r": 16, "t": 36, "b": 44},
    )
    return fig


def reference_bars(rows: Sequence[tuple[str, float]], highlight: str, x_title: str) -> go.Figure:
    """Horizontal bars comparing one result against its references."""
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": [ACCENT if label == highlight else REFERENCE for label in labels]},
            text=[f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=60 + 44 * len(rows),
        xaxis={"title": {"text": x_title}},
        yaxis={"autorange": "reversed"},
        bargap=0.35,
        margin={"l": 170, "r": 48, "t": 12, "b": 44},
    )
    return fig
