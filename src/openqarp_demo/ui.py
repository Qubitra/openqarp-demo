"""HTML building blocks for the dashboard: header bar, KPI tiles, code diff, banners."""

from __future__ import annotations

import html
import os
from collections.abc import Sequence
from dataclasses import dataclass

from qubitra.client import QubitraClient
from qubitra.errors import QubitraError

from openqarp_demo.live import CREDITS_PER_JOB, SECONDS_PER_JOB, Snapshot, Status

ENV_API_KEY = "QUBITRA_API_KEY"
ENV_API_URL = "QUBITRA_API_URL"


# -- the platform connection ---------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    """The outcome of a ``GET /v1/backends`` check."""

    ok: bool
    backends: tuple[str, ...]
    message: str


def api_url() -> str | None:
    return os.environ.get(ENV_API_URL) or None


def open_client(api_key: str) -> QubitraClient:
    """A client for the viewer's key; the key lives only in this process's memory."""
    return QubitraClient(api_key=api_key, api_url=api_url())


def check_connection(api_key: str) -> Connection:
    """List the backends the key can reach, which proves the URL and the key at once."""
    if not api_key:
        return Connection(False, (), f"No API key: set {ENV_API_KEY} or enter one above")
    try:
        with open_client(api_key) as client:
            backends = tuple(sorted(backend.id for backend in client.backends.list()))
    except QubitraError as exc:
        return Connection(False, (), f"{type(exc).__name__}: {exc.message}")
    return Connection(True, backends, f"Connected, {len(backends)} backends")


# -- markup ------------------------------------------------------------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def header(connection: Connection | None, backend: str, controls: str = "") -> str:
    if connection is None:
        dot, text = "idle", "Not connected"
    elif connection.ok:
        dot, text = "ok", f"{connection.message} · running on {backend}"
    else:
        dot, text = "bad", connection.message
    return (
        '<div class="oq-header">'
        '<div class="oq-brand">OpenQARP on Qubitra'
        "<small>Portfolio diversification with Fujitsu's OpenQARP, on a hosted simulator</small>"
        "</div>"
        '<div class="oq-header-right">'
        f'<span class="oq-status"><span class="oq-dot {dot}"></span>{_e(text)}</span>'
        f'<div class="oq-header-controls">{controls}</div>'
        "</div></div>"
    )


def kpi(label: str, value: str, note: str = "", accent: bool = False) -> str:
    cls = "oq-kpi accent" if accent else "oq-kpi"
    note_html = f'<div class="oq-kpi-note">{_e(note)}</div>' if note else ""
    return (
        f'<div class="{cls}"><div class="oq-kpi-label">{_e(label)}</div>'
        f'<div class="oq-kpi-value">{_e(value)}</div>{note_html}</div>'
    )


def kpi_grid(tiles: Sequence[str]) -> str:
    return f'<div class="oq-kpis">{"".join(tiles)}</div>'


def duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def run_kpis(snapshot: Snapshot, expected_jobs: int | None, best_label: str = "Best cut") -> str:
    of = f"of ~{expected_jobs}" if expected_jobs else ""
    best = snapshot.best
    return kpi_grid(
        [
            kpi("Jobs submitted", f"{snapshot.submitted}", of),
            kpi("Credits used", f"{snapshot.credits:,}", f"{CREDITS_PER_JOB} per job"),
            kpi("Elapsed", duration(snapshot.elapsed), "wall time"),
            kpi(best_label, "–" if best is None else f"{best:.3f}", "expected value", accent=True),
        ]
    )


def projection(jobs: int) -> str:
    return (
        f"{jobs:,} jobs · {jobs * CREDITS_PER_JOB:,} credits · "
        f"about {duration(jobs * SECONDS_PER_JOB)} at {SECONDS_PER_JOB} s per job"
    )


DIFF = (
    ("ctx", "from qarp.algorithms import QAOA"),
    ("del", "from qarp.engines import QarpEngine"),
    ("add", "from qubitra.openqarp import QubitraEngine"),
    ("ctx", ""),
    ("del", "engine = QarpEngine()"),
    ("add", 'engine = QubitraEngine("{backend}")'),
    ("ctx", ""),
    ("ctx", "qaoa = QAOA(problem=market_graph, n_layers=3, engine=engine).build()"),
    ("ctx", "energy, parameters = qaoa.run()"),
)


def code_diff(backend: str) -> str:
    marks = {"ctx": "  ", "del": "- ", "add": "+ "}
    lines = "".join(
        f'<div class="{kind}">{marks[kind]}{_e(text.format(backend=backend))}</div>'
        for kind, text in DIFF
    )
    return f'<div class="oq-diff">{lines}</div>'


def banner(snapshot: Snapshot, stage: str) -> str:
    """What the run is doing, or why it ended, in one line."""
    status = snapshot.status
    if status is Status.IDLE:
        return ""
    if status is Status.RUNNING:
        text = (
            f"<strong>{_e(stage)} is running.</strong> {snapshot.completed} jobs complete so far."
        )
    elif status is Status.FINISHED:
        text = (
            f"<strong>{_e(stage)} finished.</strong> {snapshot.submitted} jobs, "
            f"{snapshot.credits:,} credits, {duration(snapshot.elapsed)}."
        )
    elif status is Status.STOPPED:
        text = (
            f"<strong>Stopped after {snapshot.completed} jobs.</strong> The convergence so far "
            "stays on screen. Press Run to start again."
        )
    elif status is Status.DROPPED:
        in_flight = snapshot.submitted - snapshot.completed
        pending = (
            " The job in flight may still complete, and be charged, on the platform."
            if in_flight
            else ""
        )
        text = (
            f"<strong>Connection dropped after {snapshot.completed} jobs.</strong> The partial "
            f"convergence stays on screen.{pending} Nothing is resubmitted automatically, since "
            f"a retry would be a second charged job. Press Run to start again."
            f'<br><span style="color:#5b6b82;font-size:12px">{_e(snapshot.message)}</span>'
        )
    else:
        text = (
            f"<strong>{_e(stage)} stopped after {snapshot.completed} jobs.</strong> "
            f"{_e(snapshot.message)}"
        )
    return f'<div class="oq-banner {status.value}">{text}</div>'


def card(title: str, subtitle: str = "") -> str:
    sub = f'<p class="oq-sub">{_e(subtitle)}</p>' if subtitle else ""
    return f'<div class="oq-card"><h3>{_e(title)}</h3>{sub}</div>'


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{_e(c)}</td>' if i else f"<td>{_e(c)}</td>" for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="oq-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
