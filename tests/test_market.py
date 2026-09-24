"""The synthetic market, its graph and the classical references."""

from __future__ import annotations

import numpy as np

from openqarp_demo.market import (
    CORRELATION_THRESHOLD,
    N_DAYS,
    SECTORS,
    TOY_TICKERS,
    brute_force_max_cut,
    cut_value,
    diversified_basket,
    expected_cut,
    full_market,
    greedy_max_cut,
    mean_corr,
    random_cut,
    toy_market,
)


def test_full_market_has_every_sector_and_day() -> None:
    market = full_market()
    assert market.returns.shape == (N_DAYS, sum(s.n_assets for s in SECTORS.values()))
    assert market.n_assets == 50
    assert set(market.sectors) == set(SECTORS)
    assert market.tickers[0] == "TECH01"


def test_market_is_deterministic() -> None:
    assert np.array_equal(full_market().returns, full_market().returns)


def test_graph_keeps_only_correlations_above_the_threshold() -> None:
    market = full_market()
    weights = [abs(data["weight"]) for _, _, data in market.graph.edges(data=True)]
    assert min(weights) >= CORRELATION_THRESHOLD
    assert market.graph.number_of_edges() == 229
    assert sorted(market.graph.nodes) == list(range(market.n_assets))


def test_graph_edges_are_mostly_within_a_sector() -> None:
    market = full_market()
    same = sum(market.sectors[u] == market.sectors[v] for u, v in market.graph.edges())
    assert same / market.graph.number_of_edges() > 0.95


def test_toy_market_is_a_complete_graph_on_the_toy_tickers() -> None:
    toy = toy_market()
    assert toy.tickers == TOY_TICKERS
    assert toy.graph.number_of_edges() == len(TOY_TICKERS) * (len(TOY_TICKERS) - 1) // 2


def test_brute_force_beats_every_heuristic_on_the_toy_market() -> None:
    toy = toy_market()
    optimum, bits = brute_force_max_cut(toy.graph, toy.n_assets)
    assert optimum == cut_value(toy.graph, bits)
    greedy, _ = greedy_max_cut(toy.graph, toy.n_assets)
    assert greedy <= optimum + 1e-12
    assert random_cut(toy.graph, toy.n_assets) < optimum


def test_optimal_cut_splits_each_sector_pair() -> None:
    toy = toy_market()
    _, bits = brute_force_max_cut(toy.graph, toy.n_assets)
    for first in range(0, toy.n_assets, 2):
        assert bits[first] != bits[first + 1]


def test_diversified_basket_is_the_less_correlated_side() -> None:
    toy = toy_market()
    _, bits = brute_force_max_cut(toy.graph, toy.n_assets)
    basket = diversified_basket(toy.correlation, bits)
    assert sorted(basket.basket + basket.complement) == list(range(toy.n_assets))
    assert basket.basket_corr <= basket.complement_corr
    assert basket.basket_corr < mean_corr(toy.correlation, range(toy.n_assets))


def test_expected_cut_inverts_the_ising_energy() -> None:
    toy = toy_market()
    total = sum(data["weight"] for _, _, data in toy.graph.edges(data=True))
    assert expected_cut(toy.graph, -total) == total
    assert expected_cut(toy.graph, 0.0) == total / 2
