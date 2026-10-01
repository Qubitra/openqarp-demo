"""The two quantum stages, run on the Qubitra platform through OpenQARP's engine seam.

Both are OpenQARP algorithms taking a ``qarp.engines.Engine``; ``QubitraEngine`` is that
engine, so every circuit runs on the platform:

    engine = QubitraEngine("sim-statevector-26q-openqarp")

Every objective evaluation is one platform job, so the optimiser sets the bill. QAOA asks
the engine for a batched parameter-shift gradient, which puts every stencil point of a
step inside one job. PCE's optimiser is derivative-free, and each of its jobs carries one
PUB per commuting measurement group.

Adapted from OpenQARP's ``examples/use_cases/finance_portfolio_diversification.ipynb``
(Copyright 2026 Fujitsu Limited, Apache-2.0).
"""

from __future__ import annotations

import contextlib
import io
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from qarp import config
from qarp.algorithms import PCE, QAOA, Sampler, calculate_qubits
from qarp.blocks import HEABlock
from qarp.engines import Engine
from qarp.optimizers import ScipyOptimizer

from openqarp_demo.live import Tracker, instrument
from openqarp_demo.market import (
    RNG_SEED,
    Basket,
    Market,
    annualized_vol,
    brute_force_max_cut,
    cut_value,
    diversified_basket,
    expected_cut,
    greedy_max_cut,
    mean_corr,
    random_basket_corr,
    random_cut,
)

DEFAULT_BACKEND = "sim-statevector-26q-openqarp"

#: The PCE restart slider's ceiling. The notebook runs 10; a hosted copy spending a shared
#: key sets ``QUBITRA_DEMO_MAX_PCE_RESTARTS`` lower, so one visitor cannot spend the budget.
PCE_RESTARTS_CEILING = 10
ENV_MAX_PCE_RESTARTS = "QUBITRA_DEMO_MAX_PCE_RESTARTS"


def max_pce_restarts(environ: Mapping[str, str] | None = None) -> int:
    """The most PCE restarts a viewer may pick: the environment's cap, within 1..10."""
    raw = (os.environ if environ is None else environ).get(ENV_MAX_PCE_RESTARTS, "")
    try:
        cap = int(raw)
    except ValueError:
        return PCE_RESTARTS_CEILING
    return max(1, min(cap, PCE_RESTARTS_CEILING))


QAOA_LAYERS = 3
QAOA_SHOTS = 10_000
QAOA_GRADIENT = "parameter-shift"
DEFAULT_QAOA_ITERATIONS = 3

PCE_ORDER = 3
#: PCE's per-restart engine seeds count up from here, decoupled from RNG_SEED.
PCE_ENGINE_SEED = 1000

EngineFactory = Callable[[int], Engine]


def platform_engines(backend_id: str, client: Any) -> EngineFactory:
    """The platform. It seeds its own execution, so the engine takes no seed."""
    from qubitra.openqarp import QubitraEngine

    return lambda _seed: QubitraEngine(backend_id, client=client)


# -- QAOA on the 8-asset market -----------------------------------------------------


@dataclass(frozen=True)
class QaoaOutcome:
    """QAOA's answer on the small market, and the checks it is measured against."""

    energies: list[float]
    final_energy: float
    final_expected_cut: float
    bitstring: list[int]
    cut: float
    optimum: float
    optimum_bits: list[int]
    basket: Basket
    universe_corr: float
    random_corr: float
    seconds: float
    jobs: int


def run_qaoa(
    make_engine: EngineFactory,
    market: Market,
    *,
    max_iterations: int | None = DEFAULT_QAOA_ITERATIONS,
    tracker: Tracker | None = None,
) -> QaoaOutcome:
    """Max-Cut on the small market, one qubit per asset, read off by sampling the optimum."""
    tracker = tracker or Tracker()
    config.seed = RNG_SEED
    graph = market.graph
    optimum, optimum_bits = brute_force_max_cut(graph, market.n_assets)

    def as_cut(kind: str, result: Any) -> float | None:
        if kind != "objective":
            return None
        return expected_cut(graph, float(np.real(result[0])))

    started = time.perf_counter()
    options = None if max_iterations is None else {"maxiter": max_iterations}
    qaoa = QAOA(
        problem=graph,
        n_layers=QAOA_LAYERS,
        use_rzz=True,
        initial_parameters=[0.3] * 2 * QAOA_LAYERS,
        gradient=QAOA_GRADIENT,
        optimizer=ScipyOptimizer("CG", options=options),
        engine=instrument(make_engine(RNG_SEED), tracker, as_cut),
        save_energy_history=True,
    ).build()
    qaoa.suppress_success_message = True
    # OpenQARP prints "minimization did NOT finish successfully" whenever SciPy stops at the
    # iteration cap; the dashboard shows the run's state, so the print stays off the console.
    with contextlib.redirect_stdout(io.StringIO()):
        final_energy, _ = qaoa.run()

    # The state at the optimum, measured, then sampled to read off the partition.
    circuit = deepcopy(qaoa.get_final_state_block())
    circuit.measure([(qubit, qubit) for qubit in range(circuit.n_qubits)])
    sampler = Sampler(ket=circuit, n_shots=QAOA_SHOTS)
    readout = instrument(make_engine(RNG_SEED), tracker, run_kind="readout")
    readout.build([sampler])
    # The engine's return value carries the sampled distribution for this one sampler.
    probabilities: Mapping[Any, float] = readout.run({})[0]
    seconds = time.perf_counter() - started

    bits = [int(bit) for bit in max(probabilities, key=lambda key: probabilities[key])]
    basket = diversified_basket(market.correlation, bits)
    return QaoaOutcome(
        energies=[float(e) for e in qaoa.energy_history],
        final_energy=float(final_energy),
        final_expected_cut=expected_cut(graph, float(final_energy)),
        bitstring=bits,
        cut=cut_value(graph, bits),
        optimum=optimum,
        optimum_bits=optimum_bits,
        basket=basket,
        universe_corr=mean_corr(market.correlation, range(market.n_assets)),
        random_corr=random_basket_corr(market.correlation, len(basket.basket), market.n_assets),
        seconds=seconds,
        jobs=len(tracker.snapshot.points),
    )


def estimate_qaoa_jobs(max_iterations: int) -> int:
    """About how many jobs a QAOA run submits: one per objective and one per gradient for
    each iteration, plus the start and the readout. Exact to six iterations; beyond that the
    optimiser can skip a line-search step, so the run comes in a job or two under."""
    return 1 + 4 * max_iterations


# -- PCE on the 50-asset market -----------------------------------------------------


@dataclass(frozen=True)
class PceOutcome:
    """PCE's best partition of the full market, against greedy and random references."""

    n_qubits: int
    restart_cuts: list[float]
    bitstring: list[int]
    cut: float
    greedy_cut: float
    random_cut: float
    basket: Basket
    universe_corr: float
    random_corr: float
    annualized_vol: float
    sector_mix: dict[str, int]
    seconds: float
    jobs: int


def pce_qubits(market: Market) -> int:
    return int(calculate_qubits(market.n_assets, PCE_ORDER, True))


def run_pce(
    make_engine: EngineFactory,
    market: Market,
    *,
    restarts: int = 1,
    tracker: Tracker | None = None,
) -> PceOutcome:
    """The same Max-Cut on the full market, compressed into order-3 Pauli correlators."""
    tracker = tracker or Tracker()
    config.seed = RNG_SEED
    graph = market.graph
    n_qubits = pce_qubits(market)
    greedy, _ = greedy_max_cut(graph, market.n_assets)

    started = time.perf_counter()
    ansatz_rng = np.random.default_rng(RNG_SEED)
    best_value, best_solution = -np.inf, [0] * market.n_assets
    restart_cuts: list[float] = []
    for restart in range(restarts):
        ansatz = HEABlock(n_qubits, 3, True, True, True, False).build()
        initial = list(ansatz_rng.uniform(0, 2 * np.pi, len(ansatz.symbols)))
        pce = PCE(
            graph=graph,
            order=PCE_ORDER,
            ket=ansatz,
            primitive=Sampler(),
            merging=True,
            initial_parameters=initial,
            verbose=False,
            optimizer=ScipyOptimizer("COBYQA"),
            engine=instrument(make_engine(PCE_ENGINE_SEED + restart), tracker),
        ).build()
        _report_pce_cuts(pce, tracker)
        _, _, solution = pce.run()
        value = cut_value(graph, solution)
        restart_cuts.append(value)
        if value > best_value:
            best_value, best_solution = value, [int(bit) for bit in solution]
    seconds = time.perf_counter() - started

    basket = diversified_basket(market.correlation, best_solution)
    return PceOutcome(
        n_qubits=n_qubits,
        restart_cuts=restart_cuts,
        bitstring=best_solution,
        cut=float(best_value),
        greedy_cut=greedy,
        random_cut=random_cut(graph, market.n_assets),
        basket=basket,
        universe_corr=mean_corr(market.correlation, range(market.n_assets)),
        random_corr=random_basket_corr(market.correlation, len(basket.basket), market.n_assets),
        annualized_vol=annualized_vol(market.returns, basket.basket),
        sector_mix=dict(Counter(market.sectors[i] for i in basket.basket)),
        seconds=seconds,
        jobs=len(tracker.snapshot.points),
    )


def _report_pce_cuts(pce: Any, tracker: Tracker) -> None:
    """Report the cut of the partition each PCE evaluation decoded.

    PCE decodes its partition after the engine returns, so the value is attached to the
    job once the objective has been computed.
    """
    build_objective = pce.get_objective_function

    def reporting_objective() -> Callable[[Any], float]:
        objective = build_objective()

        def f(x: Any) -> float:
            loss = float(objective(x))
            tracker.set_last_value(cut_value(pce.graph, pce._last_solution))
            return loss

        return f

    pce.get_objective_function = reporting_objective


#: Measured PCE job counts on the 50-asset market: the first restart, and the mean of each
#: further one across the notebook's ten.
PCE_FIRST_RESTART_JOBS = 283
PCE_FURTHER_RESTART_JOBS = 386


def estimate_pce_jobs(restarts: int) -> int:
    """About how many jobs a PCE run submits. Later restarts start from other seeds and
    take more or fewer steps, so beyond the first this is an average."""
    return PCE_FIRST_RESTART_JOBS + PCE_FURTHER_RESTART_JOBS * (restarts - 1)
