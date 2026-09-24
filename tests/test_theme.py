"""The Plotly template and the app stylesheet load and apply."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from openqarp_demo import charts
from openqarp_demo.live import JobPoint
from openqarp_demo.market import TOY_TICKERS, full_market, toy_market
from openqarp_demo.theme import (
    ACCENT,
    SECTOR_COLORS,
    TEMPLATE_NAME,
    plotly_template,
    register_template,
    stylesheet,
)


def test_template_registers_as_the_default() -> None:
    assert register_template() == TEMPLATE_NAME
    assert pio.templates.default == TEMPLATE_NAME
    assert "Inter" in plotly_template().layout.font.family


def test_stylesheet_ships_with_the_package() -> None:
    css = stylesheet()
    assert "fonts.googleapis.com" in css
    assert ".oq-header" in css
    assert ".oq-kpi" in css


def test_accent_is_reserved_for_the_basket() -> None:
    assert ACCENT not in SECTOR_COLORS.values()


def test_market_network_draws_every_asset() -> None:
    market = full_market()
    fig = charts.market_network(market, highlight=TOY_TICKERS)
    assert isinstance(fig, go.Figure)
    sectors = [t for t in fig.data if t.name in SECTOR_COLORS]
    assert sum(len(t.x) for t in sectors) == market.n_assets
    ringed = [
        text
        for t in sectors
        for text, width in zip(t.text, t.marker.line.width, strict=True)
        if width == 3
    ]
    assert sorted(ringed) == sorted(TOY_TICKERS)


def test_partition_graph_colours_the_basket_with_the_accent() -> None:
    toy = toy_market()
    fig = charts.partition_graph(toy, [1, 3, 5, 6])
    basket = next(t for t in fig.data if t.name == "Diversified basket")
    assert basket.marker.color == ACCENT
    assert len(basket.x) == 4


def test_convergence_plots_only_valued_jobs() -> None:
    points = [
        JobPoint(1, "objective", 1.0, 0.1),
        JobPoint(2, "gradient", None, 0.2),
        JobPoint(3, "objective", 2.0, 0.3),
    ]
    fig = charts.convergence(points, reference=3.0, reference_label="Optimum", expected_jobs=13)
    assert list(fig.data[0].x) == [1, 3]
    assert list(fig.data[1].y) == [1.0, 2.0]
    assert fig.layout.xaxis.range == (0.5, 13.5)
