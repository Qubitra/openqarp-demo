"""The app's compute path, headless, with OpenQARP's local engine standing in for the
platform: no network."""

from __future__ import annotations

import pytest
from qarp.engines import QarpEngine

from openqarp_demo.live import Status, Tracker
from openqarp_demo.market import Market, full_market, toy_market
from openqarp_demo.runs import (
    EngineFactory,
    estimate_pce_jobs,
    estimate_qaoa_jobs,
    max_pce_restarts,
    pce_qubits,
    run_pce,
    run_qaoa,
)


def local_engines() -> EngineFactory:
    """The platform's stand-in: OpenQARP's own simulator, seeded per engine."""
    return lambda seed: QarpEngine(seed=seed)


@pytest.fixture(scope="module")
def market() -> Market:
    return full_market()


def test_capped_qaoa_reaches_the_brute_force_optimum(market: Market) -> None:
    toy = toy_market(market)
    tracker = Tracker()
    tracker.start()
    outcome = run_qaoa(local_engines(), toy, max_iterations=3, tracker=tracker)
    tracker.finish(outcome)

    assert outcome.jobs == 13
    assert outcome.cut == pytest.approx(outcome.optimum)
    assert outcome.basket.basket_corr < outcome.universe_corr
    kinds = [p.kind for p in tracker.snapshot.points]
    assert kinds.count("readout") == 1
    assert kinds.count("gradient") == 6
    assert tracker.snapshot.status is Status.FINISHED
    # The convergence trace ends where the optimiser did.
    assert tracker.snapshot.values[-1][1] == pytest.approx(outcome.final_expected_cut)


@pytest.mark.parametrize("iterations", [1, 3, 6, 10])
def test_the_qaoa_estimate_matches_the_run_to_two_jobs(market: Market, iterations: int) -> None:
    jobs = run_qaoa(local_engines(), toy_market(market), max_iterations=iterations).jobs
    estimate = estimate_qaoa_jobs(iterations)
    assert estimate - 2 <= jobs <= estimate
    if iterations <= 6:
        assert jobs == estimate


def test_one_pce_restart_beats_a_random_split(market: Market) -> None:
    tracker = Tracker()
    tracker.start()
    outcome = run_pce(local_engines(), market, restarts=1, tracker=tracker)

    assert outcome.n_qubits == pce_qubits(market) == 5
    assert outcome.cut > outcome.random_cut
    assert outcome.cut <= outcome.greedy_cut + 1e-9
    assert outcome.jobs == len(tracker.snapshot.points) == estimate_pce_jobs(1)
    assert all(p.value is not None for p in tracker.snapshot.points)
    assert sum(outcome.sector_mix.values()) == len(outcome.basket.basket)


def test_pce_restarts_default_to_the_notebooks_ten() -> None:
    assert max_pce_restarts({}) == 10


def test_pce_restarts_follow_the_environment_cap() -> None:
    assert max_pce_restarts({"QUBITRA_DEMO_MAX_PCE_RESTARTS": "2"}) == 2


def test_pce_restarts_cap_is_clamped_and_ignores_junk() -> None:
    assert max_pce_restarts({"QUBITRA_DEMO_MAX_PCE_RESTARTS": "0"}) == 1
    assert max_pce_restarts({"QUBITRA_DEMO_MAX_PCE_RESTARTS": "50"}) == 10
    assert max_pce_restarts({"QUBITRA_DEMO_MAX_PCE_RESTARTS": "lots"}) == 10
