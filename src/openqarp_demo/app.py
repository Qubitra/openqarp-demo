import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="OpenQARP on Qubitra",
    css_file="theme.css",
)


@app.cell(hide_code=True)
def _():
    import os

    import marimo as mo

    from openqarp_demo import charts, ui
    from openqarp_demo.live import JobCounter, Snapshot, Status, Tracker
    from openqarp_demo.market import TOY_TICKERS, full_market, toy_market
    from openqarp_demo.runs import (
        DEFAULT_BACKEND,
        DEFAULT_QAOA_ITERATIONS,
        local_engines,
        pce_qubits,
        platform_engines,
        project_pce_jobs,
        run_pce,
        run_qaoa,
    )

    return (
        DEFAULT_BACKEND,
        DEFAULT_QAOA_ITERATIONS,
        JobCounter,
        Snapshot,
        Status,
        TOY_TICKERS,
        Tracker,
        charts,
        full_market,
        local_engines,
        mo,
        os,
        pce_qubits,
        platform_engines,
        project_pce_jobs,
        run_pce,
        run_qaoa,
        toy_market,
        ui,
    )


@app.cell(hide_code=True)
def _(mo, os, ui):
    # The key comes from the environment, or from a password field when it is unset.
    # Either way it lives only in this process's memory: nothing writes it anywhere.
    env_key = os.environ.get(ui.ENV_API_KEY, "")
    key_input = mo.ui.text(kind="password", placeholder="qpk_…", label="API key", full_width=False)
    recheck = mo.ui.button(label="Re-check", value=0, on_click=lambda count: count + 1)
    return env_key, key_input, recheck


@app.cell(hide_code=True)
def _(env_key, key_input, recheck, ui):
    recheck.value
    api_key = env_key or key_input.value
    connection = ui.check_connection(api_key)
    return api_key, connection


@app.cell(hide_code=True)
def _(DEFAULT_BACKEND, connection, mo):
    _options = list(connection.backends) or [DEFAULT_BACKEND]
    backend = mo.ui.dropdown(
        options=_options,
        value=DEFAULT_BACKEND if DEFAULT_BACKEND in _options else _options[0],
        label="Backend",
    )
    return (backend,)


@app.cell(hide_code=True)
def _(backend, connection, env_key, key_input, mo, recheck, ui):
    _controls = [backend, recheck] if env_key else [key_input, backend, recheck]
    mo.vstack(
        [
            mo.Html(ui.header(connection, backend.value)),
            mo.hstack(_controls, justify="end", gap=1),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(backend, mo, ui):
    mo.vstack(
        [
            mo.md(
                "### One line moves OpenQARP onto the platform\n"
                "Fujitsu's OpenQARP runs its algorithms against an *engine*. Swap the local "
                "simulator for `QubitraEngine` and the same QAOA runs on Qubitra's hosted "
                "simulator; every other line is OpenQARP's own code."
            ),
            mo.Html(ui.code_diff(backend.value)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(TOY_TICKERS, full_market, toy_market):
    market = full_market()
    toy = toy_market(market)
    return market, toy


@app.cell(hide_code=True)
def _(TOY_TICKERS, charts, market, mo):
    mo.vstack(
        [
            mo.md(
                "### The market\n"
                f"{market.n_assets} synthetic assets across five sectors, 500 trading days. "
                "An edge joins two assets whose daily returns correlate by 0.15 or more; "
                "same-sector assets cluster. Splitting the graph with a **maximum cut** pushes "
                "correlated assets onto opposite sides, so each side is a diversified basket. "
                f"The {len(TOY_TICKERS)} ringed assets are the market QAOA solves live, "
                "one qubit per asset."
            ),
            charts.market_network(market, highlight=TOY_TICKERS),
        ]
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_QAOA_ITERATIONS, mo):
    iterations = mo.ui.slider(
        1, 10, value=DEFAULT_QAOA_ITERATIONS, label="Optimiser iterations", show_value=True
    )
    return (iterations,)


@app.cell(hide_code=True)
def _(Tracker, iterations, local_engines, run_qaoa, toy):
    # The same QAOA on a laptop: it is the side-by-side reference, and counting its engine
    # calls gives the platform run's job count before anything is submitted.
    local_tracker = Tracker()
    local_qaoa = run_qaoa(
        local_engines(), toy, max_iterations=iterations.value, tracker=local_tracker
    )
    return local_qaoa, local_tracker


@app.cell(hide_code=True)
def _(Snapshot, mo):
    get_qaoa, set_qaoa = mo.state(Snapshot())
    qaoa_control = {"tracker": None}
    return get_qaoa, qaoa_control, set_qaoa


@app.cell(hide_code=True)
def _(connection, mo, qaoa_control):
    def _stop(count):
        if qaoa_control["tracker"] is not None:
            qaoa_control["tracker"].stop()
        return count + 1

    qaoa_run = mo.ui.run_button(
        label="Run QAOA on the platform", kind="success", disabled=not connection.ok
    )
    qaoa_stop = mo.ui.button(label="Stop", value=0, on_click=_stop, kind="danger")
    return qaoa_run, qaoa_stop


@app.cell(hide_code=True)
def _(iterations, local_qaoa, mo, qaoa_run, qaoa_stop, ui):
    mo.vstack(
        [
            mo.md(
                "### Live optimisation\n"
                "QAOA on the 8-asset market, three layers. Each point is one platform job: "
                "an objective evaluation, with the batched parameter-shift gradient riding "
                "a job of its own between them."
            ),
            mo.hstack(
                [
                    iterations,
                    mo.md(f"Projected: **{ui.projection(local_qaoa.jobs)}**"),
                    qaoa_run,
                    qaoa_stop,
                ],
                justify="start",
                align="center",
                gap=1.5,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(
    JobCounter,
    Tracker,
    api_key,
    backend,
    iterations,
    mo,
    platform_engines,
    qaoa_control,
    qaoa_run,
    run_qaoa,
    set_qaoa,
    toy,
    ui,
):
    mo.stop(not qaoa_run.value)

    def _work(tracker, backend_id, max_iterations):
        try:
            client = tracker.counter.wrap(ui.open_client(api_key))
        except Exception as exc:
            tracker.fail(exc)
            return
        try:
            tracker.start()
            outcome = run_qaoa(
                platform_engines(backend_id, client),
                toy,
                max_iterations=max_iterations,
                tracker=tracker,
            )
            tracker.finish(outcome)
        except Exception as exc:  # the dashboard shows why; nothing is resubmitted
            tracker.fail(exc)
        finally:
            client.close()

    _tracker = Tracker(listener=set_qaoa, counter=JobCounter())
    qaoa_control["tracker"] = _tracker
    mo.Thread(target=_work, args=(_tracker, backend.value, iterations.value), daemon=True).start()
    return


@app.cell(hide_code=True)
def _(charts, get_qaoa, local_qaoa, mo, ui):
    qaoa_snapshot = get_qaoa()
    mo.vstack(
        [
            mo.Html(ui.banner(qaoa_snapshot, "QAOA")),
            mo.hstack(
                [
                    charts.convergence(
                        qaoa_snapshot.points,
                        reference=local_qaoa.optimum,
                        reference_label="Brute-force optimum",
                        expected_jobs=local_qaoa.jobs,
                    ),
                    mo.Html(ui.run_kpis(qaoa_snapshot, local_qaoa.jobs, "Best expected cut")),
                ],
                widths=[2, 1],
                align="center",
            ),
        ]
    )
    return (qaoa_snapshot,)


@app.cell(hide_code=True)
def _(Status, charts, mo, qaoa_snapshot, toy, ui):
    mo.stop(qaoa_snapshot.status is not Status.FINISHED)
    _o = qaoa_snapshot.outcome
    _names = ", ".join(toy.tickers[i] for i in _o.basket.basket)
    mo.vstack(
        [
            mo.md(
                "### Result\n"
                f"The most likely bitstring of the optimised state splits the market into two "
                f"baskets of {len(_o.basket.basket)} and {len(_o.basket.complement)}. "
                f"The diversified basket is **{_names}**."
            ),
            mo.hstack(
                [
                    charts.partition_graph(toy, _o.basket.basket),
                    mo.Html(
                        ui.kpi_grid(
                            [
                                ui.kpi("QAOA cut", f"{_o.cut:.3f}", "sampled bitstring", True),
                                ui.kpi(
                                    "Brute-force optimum",
                                    f"{_o.optimum:.3f}",
                                    f"ratio {_o.cut / _o.optimum:.3f}",
                                ),
                                ui.kpi(
                                    "Basket correlation",
                                    f"{_o.basket.basket_corr:+.3f}",
                                    "mean pairwise",
                                    True,
                                ),
                                ui.kpi(
                                    "Whole market",
                                    f"{_o.universe_corr:+.3f}",
                                    f"random basket {_o.random_corr:+.3f}",
                                ),
                            ]
                        )
                    ),
                ],
                widths=[3, 2],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(Status, charts, local_qaoa, local_tracker, mo, qaoa_snapshot, ui):
    mo.stop(qaoa_snapshot.status is not Status.FINISHED)
    _p = qaoa_snapshot.outcome
    _local_values = [p.value for p in local_tracker.snapshot.points]
    _platform_values = [p.value for p in qaoa_snapshot.points]
    _deltas = [
        abs(a - b)
        for a, b in zip(_local_values, _platform_values, strict=False)
        if a is not None and b is not None
    ]
    _max_delta = max(_deltas, default=0.0)
    mo.vstack(
        [
            mo.md(
                "### Laptop against platform\n"
                "The same QAOA, the same seeds, run on OpenQARP's local `QarpEngine` and on "
                f"the platform. The largest per-job difference in the objective is "
                f"**{_max_delta:.1e}**."
            ),
            mo.hstack(
                [
                    charts.side_by_side(
                        local_tracker.snapshot.points, qaoa_snapshot.points, local_qaoa.optimum
                    ),
                    mo.Html(
                        ui.table(
                            ["", "Local QarpEngine", "Qubitra platform"],
                            [
                                [
                                    "Final expected cut",
                                    f"{local_qaoa.final_expected_cut:.6f}",
                                    f"{_p.final_expected_cut:.6f}",
                                ],
                                ["Sampled cut", f"{local_qaoa.cut:.3f}", f"{_p.cut:.3f}"],
                                ["Jobs", f"{local_qaoa.jobs}", f"{qaoa_snapshot.submitted}"],
                                ["Credits", "0", f"{qaoa_snapshot.credits:,}"],
                                [
                                    "Wall time",
                                    ui.duration(local_qaoa.seconds),
                                    ui.duration(_p.seconds),
                                ],
                            ],
                        )
                    ),
                ],
                widths=[3, 2],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(market, mo, pce_qubits):
    pce_restarts = mo.ui.slider(1, 10, value=1, label="PCE restarts", show_value=True)
    pce_estimate = mo.ui.run_button(label="Estimate the cost")
    mo.vstack(
        [
            mo.md(
                "### Optional: all 50 assets with PCE\n"
                f"Pauli Correlation Encoding fits the full {market.n_assets}-asset Max-Cut into "
                f"**{pce_qubits(market)} qubits** using order-3 Pauli correlators. It is a "
                "longer run, so the cost is projected before anything is submitted."
            ),
            mo.hstack([pce_restarts, pce_estimate], justify="start", align="center", gap=1.5),
        ]
    )
    return pce_estimate, pce_restarts


@app.cell(hide_code=True)
def _(market, mo, pce_estimate, pce_restarts, project_pce_jobs):
    mo.stop(not pce_estimate.value)
    with mo.status.spinner(title="Counting the jobs locally…"):
        pce_plan = {
            "restarts": pce_restarts.value,
            "jobs": project_pce_jobs(market, pce_restarts.value),
        }
    return (pce_plan,)


@app.cell(hide_code=True)
def _(Snapshot, mo):
    get_pce, set_pce = mo.state(Snapshot())
    pce_control = {"tracker": None}
    return get_pce, pce_control, set_pce


@app.cell(hide_code=True)
def _(connection, mo, pce_control, pce_plan, ui):
    def _stop(count):
        if pce_control["tracker"] is not None:
            pce_control["tracker"].stop()
        return count + 1

    pce_run = mo.ui.run_button(
        label="Start PCE on the platform", kind="warn", disabled=not connection.ok
    )
    pce_stop = mo.ui.button(label="Stop", value=0, on_click=_stop, kind="danger")
    mo.hstack(
        [
            mo.md(
                f"Projected for {pce_plan['restarts']} restart(s): "
                f"**{ui.projection(pce_plan['jobs'])}**"
            ),
            pce_run,
            pce_stop,
        ],
        justify="start",
        align="center",
        gap=1.5,
        wrap=True,
    )
    return pce_run, pce_stop


@app.cell
def _(
    JobCounter,
    Tracker,
    api_key,
    backend,
    market,
    mo,
    pce_control,
    pce_plan,
    pce_run,
    platform_engines,
    run_pce,
    set_pce,
    ui,
):
    mo.stop(not pce_run.value)

    def _work(tracker, backend_id, restarts):
        try:
            client = tracker.counter.wrap(ui.open_client(api_key))
        except Exception as exc:
            tracker.fail(exc)
            return
        try:
            tracker.start()
            outcome = run_pce(
                platform_engines(backend_id, client), market, restarts=restarts, tracker=tracker
            )
            tracker.finish(outcome)
        except Exception as exc:  # the dashboard shows why; nothing is resubmitted
            tracker.fail(exc)
        finally:
            client.close()

    _tracker = Tracker(listener=set_pce, counter=JobCounter())
    pce_control["tracker"] = _tracker
    mo.Thread(
        target=_work, args=(_tracker, backend.value, pce_plan["restarts"]), daemon=True
    ).start()
    return


@app.cell(hide_code=True)
def _(Status, charts, get_pce, mo, pce_plan, ui):
    pce_snapshot = get_pce()
    mo.stop(pce_snapshot.status is Status.IDLE)
    mo.vstack(
        [
            mo.Html(ui.banner(pce_snapshot, "PCE")),
            mo.hstack(
                [
                    charts.convergence(
                        pce_snapshot.points,
                        reference=None,
                        reference_label="",
                        expected_jobs=pce_plan["jobs"],
                        y_title="Cut of the decoded partition",
                    ),
                    mo.Html(ui.run_kpis(pce_snapshot, pce_plan["jobs"], "Best cut")),
                ],
                widths=[2, 1],
                align="center",
            ),
        ]
    )
    return (pce_snapshot,)


@app.cell(hide_code=True)
def _(Status, charts, market, mo, pce_snapshot, ui):
    mo.stop(pce_snapshot.status is not Status.FINISHED)
    _o = pce_snapshot.outcome
    _mix = ", ".join(f"{sector} {count}" for sector, count in sorted(_o.sector_mix.items()))
    mo.vstack(
        [
            mo.md(
                f"### PCE result\n"
                f"A **{len(_o.basket.basket)}-asset basket** drawn from every sector ({_mix}), "
                f"annualised volatility {_o.annualized_vol * 100:.2f}%."
            ),
            mo.hstack(
                [
                    charts.partition_graph(market, _o.basket.basket, labels=False),
                    mo.vstack(
                        [
                            mo.md("**Cut value**"),
                            charts.reference_bars(
                                [
                                    ("Random split", _o.random_cut),
                                    ("PCE, best restart", _o.cut),
                                    ("Greedy local search", _o.greedy_cut),
                                ],
                                "PCE, best restart",
                                "Max-Cut objective",
                            ),
                            mo.md("**Mean pairwise correlation**"),
                            charts.reference_bars(
                                [
                                    ("PCE basket", _o.basket.basket_corr),
                                    ("Random basket, same size", _o.random_corr),
                                    ("Whole market", _o.universe_corr),
                                ],
                                "PCE basket",
                                "Correlation",
                            ),
                        ]
                    ),
                ],
                widths=[1, 1],
                align="center",
            ),
            mo.Html(
                ui.table(
                    ["", "Value"],
                    [
                        ["Qubits", _o.n_qubits],
                        ["Restarts", len(_o.restart_cuts)],
                        ["Jobs", pce_snapshot.submitted],
                        ["Credits", f"{pce_snapshot.credits:,}"],
                        ["Wall time", ui.duration(_o.seconds)],
                    ],
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        "<small>The market model, QAOA and PCE workflows are adapted from OpenQARP's "
        "portfolio diversification notebook (Copyright 2026 Fujitsu Limited, Apache-2.0).</small>"
    )
    return


if __name__ == "__main__":
    app.run()
