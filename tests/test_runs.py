"""The app's compute path, headless, on OpenQARP's local engine: no network."""

from __future__ import annotations

import pytest

from openqarp_demo.live import Status, Tracker
from openqarp_demo.market import Market, full_market, toy_market
from openqarp_demo.runs import (
    local_engines,
    max_pce_restarts,
    pce_qubits,
    project_qaoa_jobs,
    run_pce,
    run_qaoa,
)


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


@pytest.mark.parametrize(("iterations", "jobs"), [(1, 5), (2, 9), (3, 13)])
def test_projection_counts_four_jobs_per_iteration_plus_readout(
    market: Market, iterations: int, jobs: int
) -> None:
    assert project_qaoa_jobs(toy_market(market), iterations) == jobs


def test_one_pce_restart_beats_a_random_split(market: Market) -> None:
    tracker = Tracker()
    tracker.start()
    outcome = run_pce(local_engines(), market, restarts=1, tracker=tracker)

    assert outcome.n_qubits == pce_qubits(market) == 5
    assert outcome.cut > outcome.random_cut
    assert outcome.cut <= outcome.greedy_cut + 1e-9
    assert outcome.jobs == len(tracker.snapshot.points) > 100
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
