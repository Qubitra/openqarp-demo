"""The synthetic market, its graph, and the classical references the quantum runs meet.

A market graph turns pairwise return correlation into Max-Cut edge weights: maximising the
cut pushes strongly correlated assets onto opposite sides, so each side of the partition is
a more mutually diversified basket than a same-sized random pick.

Adapted from OpenQARP's ``examples/use_cases/finance_portfolio_diversification.ipynb``
(Copyright 2026 Fujitsu Limited, Apache-2.0), with the same seeds and market model.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qarp.algorithms import classical_function_max_cut
from qarp.graphs import Graph

#: One seed for the whole notebook, as the source notebook has it.
RNG_SEED = 42
N_DAYS = 500
#: Correlations below this in magnitude are dropped from the full market graph: at 0.15
#: only genuinely same-sector pairs survive, so the graph is close to five near-cliques.
CORRELATION_THRESHOLD = 0.15

#: Two tickers from each of four sectors: the small market QAOA runs on, one qubit each.
TOY_TICKERS = (
    "TECH01",
    "TECH02",
    "ENRG01",
    "ENRG02",
    "FINL01",
    "FINL02",
    "GOLD01",
    "GOLD02",
)


@dataclass(frozen=True)
class Sector:
    """A sector's share of the synthetic universe, in a multi-factor return model."""

    n_assets: int
    beta: float
    sector_vol: float
    idio_vol: float


#: Fifty tickers across five sectors. Each asset's return is driven mostly by its sector
#: factor plus a much smaller shared market factor (sign-flipped for GOLD) and
#: idiosyncratic noise, so correlation concentrates within a sector.
SECTORS: Mapping[str, Sector] = {
    "TECH": Sector(n_assets=12, beta=0.15, sector_vol=0.014, idio_vol=0.006),
    "ENRG": Sector(n_assets=10, beta=0.15, sector_vol=0.013, idio_vol=0.006),
    "FINL": Sector(n_assets=10, beta=0.15, sector_vol=0.013, idio_vol=0.006),
    "HLTH": Sector(n_assets=10, beta=0.15, sector_vol=0.012, idio_vol=0.006),
    "GOLD": Sector(n_assets=8, beta=-0.10, sector_vol=0.012, idio_vol=0.006),
}


@dataclass(frozen=True)
class Market:
    """Daily returns for a set of assets, with their correlation and Max-Cut graph."""

    returns: np.ndarray
    tickers: tuple[str, ...]
    sector_of: Mapping[str, str]
    correlation: np.ndarray
    graph: Graph

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def sectors(self) -> list[str]:
        return [self.sector_of[ticker] for ticker in self.tickers]

    def subset(self, tickers: Sequence[str], threshold: float = 0.0) -> Market:
        """The same market restricted to ``tickers``, renumbered from node 0."""
        columns = [self.tickers.index(ticker) for ticker in tickers]
        returns = self.returns[:, columns]
        graph, correlation = market_graph(returns, threshold)
        return Market(
            returns=returns,
            tickers=tuple(tickers),
            sector_of={ticker: self.sector_of[ticker] for ticker in tickers},
            correlation=correlation,
            graph=graph,
        )


def simulate_returns() -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Daily returns for the whole universe, as a (days, assets) array."""
    rng = np.random.default_rng(RNG_SEED)
    market_factor = rng.normal(0.0003, 0.006, N_DAYS)
    tickers: list[str] = []
    columns: list[np.ndarray] = []
    sector_of: dict[str, str] = {}
    for name, sector in SECTORS.items():
        sector_factor = rng.normal(0.0, sector.sector_vol, N_DAYS)
        for index in range(sector.n_assets):
            ticker = f"{name}{index + 1:02d}"
            idiosyncratic = rng.normal(0.0, sector.idio_vol, N_DAYS)
            columns.append(sector.beta * market_factor + sector_factor + idiosyncratic)
            tickers.append(ticker)
            sector_of[ticker] = name
    return np.array(columns).T, tickers, sector_of


def market_graph(returns: np.ndarray, threshold: float) -> tuple[Graph, np.ndarray]:
    """Assets as nodes, pairwise return correlation as edge weights.

    QAOA and PCE both read node labels as qubit indices, so asset ``i`` is node ``i``.
    """
    correlation = np.corrcoef(returns, rowvar=False)
    n_assets = returns.shape[1]
    graph = Graph()
    graph.add_nodes_from(range(n_assets))
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            weight = float(correlation[i, j])
            if abs(weight) >= threshold:
                graph.add_edge(i, j, weight=weight)
    return graph, correlation


def full_market() -> Market:
    """The 50-asset universe with its thresholded market graph."""
    returns, tickers, sector_of = simulate_returns()
    graph, correlation = market_graph(returns, CORRELATION_THRESHOLD)
    return Market(
        returns=returns,
        tickers=tuple(tickers),
        sector_of=sector_of,
        correlation=correlation,
        graph=graph,
    )


def toy_market(market: Market | None = None) -> Market:
    """The 8-asset market QAOA runs on: every pair is an edge, small enough to check."""
    return (market or full_market()).subset(TOY_TICKERS, threshold=0.0)


# -- diversification metrics --------------------------------------------------------


def mean_corr(correlation: np.ndarray, indices: Sequence[int]) -> float:
    """Average pairwise correlation over a subset of assets."""
    if len(indices) < 2:
        return float("nan")
    block = correlation[np.ix_(list(indices), list(indices))]
    upper = np.triu_indices(len(indices), k=1)
    return float(block[upper].mean())


@dataclass(frozen=True)
class Basket:
    """The two sides of a Max-Cut partition, the less mutually correlated side first."""

    basket: list[int]
    complement: list[int]
    basket_corr: float
    complement_corr: float


def diversified_basket(correlation: np.ndarray, bitstring: Sequence[int]) -> Basket:
    """Which side of a cut is the diversified basket, read off rather than assumed.

    Max-Cut only guarantees the correlation left uncut is small, not that it splits
    evenly between the two sides.
    """
    side0 = [i for i, bit in enumerate(bitstring) if bit == 0]
    side1 = [i for i, bit in enumerate(bitstring) if bit == 1]
    corr0, corr1 = mean_corr(correlation, side0), mean_corr(correlation, side1)
    if corr0 <= corr1:
        return Basket(side0, side1, corr0, corr1)
    return Basket(side1, side0, corr1, corr0)


def random_basket_corr(
    correlation: np.ndarray, size: int, n_assets: int, trials: int = 2000
) -> float:
    """Average pairwise correlation of a random basket of the same size."""
    rng = np.random.default_rng(0)
    return float(
        np.mean(
            [
                mean_corr(correlation, [int(i) for i in rng.choice(n_assets, size, replace=False)])
                for _ in range(trials)
            ]
        )
    )


def annualized_vol(returns: np.ndarray, indices: Sequence[int], trading_days: int = 252) -> float:
    """Annualised volatility of an equal-weight basket."""
    weights = np.ones(len(indices)) / len(indices)
    covariance = np.cov(returns[:, list(indices)], rowvar=False)
    return float(np.sqrt((weights @ covariance @ weights) * trading_days))


def cut_value(graph: Graph, bitstring: Sequence[int]) -> float:
    """The Max-Cut objective of a partition."""
    return float(classical_function_max_cut(graph, list(bitstring)))


def total_weight(graph: Graph) -> float:
    """The summed edge weight, which turns QAOA's Ising energy into an expected cut."""
    return float(sum(data.get("weight", 1.0) for _, _, data in graph.edges(data=True)))


def expected_cut(graph: Graph, energy: float) -> float:
    """``cut = (sum of weights - <H>) / 2`` for QAOA's Ising cost Hamiltonian."""
    return (total_weight(graph) - energy) / 2


def brute_force_max_cut(graph: Graph, n_nodes: int) -> tuple[float, list[int]]:
    """The true optimum, for a market small enough to enumerate."""
    best_value, best_bits = -np.inf, [0] * n_nodes
    for bits in itertools.product([0, 1], repeat=n_nodes):
        value = cut_value(graph, bits)
        if value > best_value:
            best_value, best_bits = value, list(bits)
    return float(best_value), best_bits


def greedy_max_cut(graph: Graph, n_nodes: int, n_restarts: int = 20) -> tuple[float, list[int]]:
    """Repeated single-asset flips from several random starts: the classical reference."""
    rng = np.random.default_rng(1)
    best_value, best_bits = -np.inf, [0] * n_nodes
    for _ in range(n_restarts):
        bits = [int(bit) for bit in rng.integers(0, 2, n_nodes)]
        value = cut_value(graph, bits)
        improved = True
        while improved:
            improved = False
            for i in range(n_nodes):
                bits[i] ^= 1
                candidate = cut_value(graph, bits)
                if candidate > value:
                    value = candidate
                    improved = True
                else:
                    bits[i] ^= 1
        if value > best_value:
            best_value, best_bits = value, bits[:]
    return float(best_value), best_bits


def random_cut(graph: Graph, n_nodes: int, trials: int = 300) -> float:
    """The average cut of a purely random split."""
    rng = np.random.default_rng(0)
    return float(
        np.mean(
            [
                cut_value(graph, [int(bit) for bit in rng.integers(0, 2, n_nodes)])
                for _ in range(trials)
            ]
        )
    )
