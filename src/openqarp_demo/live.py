"""Live progress of a run: the job counter, the per-job trace and the run's outcome state.

Every job the platform engine submits passes through ``QubitraClient.jobs.run``, so a
wrapper there counts platform jobs exactly. The engine itself is instrumented too: each
``run``, ``run_gradient`` or ``batch_run`` call is one job on the platform, and records the
objective value that job produced.

A :class:`Tracker` holds that trace behind a lock and hands an immutable
:class:`Snapshot` to a listener after every change, which is what the dashboard renders.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from qubitra.client import QubitraClient
from qubitra.errors import ProviderError, QubitraError

#: What a job costs and how long it takes on the hosted simulator, for projections.
#: The time is the capped QAOA stage measured end to end with qubitra-sdk 0.4.3.
CREDITS_PER_JOB = 10
SECONDS_PER_JOB = 1.2


class Status(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"
    DROPPED = "dropped"
    FAILED = "failed"


class Stopped(Exception):
    """The presenter pressed Stop; raised before the next job is submitted."""


@dataclass(frozen=True)
class JobPoint:
    """One completed job: its sequence number, what it computed, and when it finished."""

    job: int
    kind: str
    value: float | None
    elapsed: float


@dataclass(frozen=True)
class Snapshot:
    """An immutable view of a run, safe to hand across threads."""

    status: Status = Status.IDLE
    submitted: int = 0
    completed: int = 0
    pubs: int = 0
    elapsed: float = 0.0
    points: tuple[JobPoint, ...] = ()
    message: str = ""
    outcome: Any = None

    @property
    def credits(self) -> int:
        return self.submitted * CREDITS_PER_JOB

    @property
    def values(self) -> list[tuple[int, float]]:
        return [(p.job, p.value) for p in self.points if p.value is not None]

    @property
    def best(self) -> float | None:
        values = [value for _, value in self.values]
        return max(values) if values else None


Listener = Callable[[Snapshot], None]


class JobCounter:
    """Counts the jobs and PUBs a client submits, at its one public submission seam."""

    def __init__(self, on_submit: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._on_submit = on_submit
        # Called after a submission is counted, so a dashboard can show the job in flight.
        self.on_change: Callable[[], None] | None = None
        self.submitted = 0
        self.completed = 0
        self.pubs = 0

    def wrap(self, client: QubitraClient) -> QubitraClient:
        """Wrap ``client.jobs.run`` so every submission is counted, and return the client.

        A submission is counted before it leaves: a request whose response is lost may
        still run, and be charged, on the platform.
        """
        submit = client.jobs.run

        def counted_run(*, pubs: Sequence[Any], **kwargs: Any) -> Any:
            if self._on_submit is not None:
                self._on_submit()
            with self._lock:
                self.submitted += 1
                self.pubs += len(pubs)
            if self.on_change is not None:
                self.on_change()
            job = submit(pubs=pubs, **kwargs)
            with self._lock:
                self.completed += 1
            return job

        client.jobs.run = counted_run  # type: ignore[method-assign]
        return client


def _is_transport_failure(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and "transport" in str(exc).lower()


@dataclass
class Tracker:
    """The live trace of one run, reported to ``listener`` after every change."""

    listener: Listener | None = None
    counter: JobCounter | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _snapshot: Snapshot = field(default_factory=Snapshot)
    _started: float = 0.0
    _engine_jobs: int = 0

    def __post_init__(self) -> None:
        if self.counter is not None and self.counter.on_change is None:
            self.counter.on_change = self._refresh

    @property
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def _refresh(self) -> None:
        """Re-read the counts and redraw; a job has just been submitted."""
        with self._lock:
            if self._snapshot.status is not Status.RUNNING:
                return
            self._snapshot = self._counts(self._snapshot)
        self._notify()

    def start(self) -> None:
        with self._lock:
            self._started = time.perf_counter()
            self._engine_jobs = 0
            self._snapshot = Snapshot(status=Status.RUNNING)
        self._notify()

    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped

    def stop(self) -> None:
        self.stop_event.set()

    def record_job(self, kind: str, value: float | None) -> None:
        """A job finished; ``value`` is the objective it produced, if it produced one."""
        with self._lock:
            self._engine_jobs += 1
            point = JobPoint(self._engine_jobs, kind, value, self._elapsed())
            self._snapshot = self._counts(
                replace(self._snapshot, points=(*self._snapshot.points, point))
            )
        self._notify()

    def set_last_value(self, value: float) -> None:
        """Attach an objective value to the latest job, once the caller has derived it."""
        with self._lock:
            points = self._snapshot.points
            if not points:
                return
            last = replace(points[-1], value=value)
            self._snapshot = replace(self._snapshot, points=(*points[:-1], last))
        self._notify()

    def finish(self, outcome: Any) -> None:
        self._end(Status.FINISHED, "", outcome)

    def fail(self, exc: BaseException) -> None:
        """Record why a run ended early. Nothing is resubmitted: a retry is a new charge."""
        if isinstance(exc, Stopped):
            self._end(Status.STOPPED, "Stopped by the presenter.", None)
        elif _is_transport_failure(exc):
            self._end(Status.DROPPED, str(exc), None)
        elif isinstance(exc, QubitraError):
            self._end(Status.FAILED, f"{type(exc).__name__}: {exc}", None)
        else:
            self._end(Status.FAILED, f"{type(exc).__name__}: {exc}", None)

    def _end(self, status: Status, message: str, outcome: Any) -> None:
        with self._lock:
            self._snapshot = self._counts(
                replace(self._snapshot, status=status, message=message, outcome=outcome)
            )
        self._notify()

    def _counts(self, snapshot: Snapshot) -> Snapshot:
        """Platform counts when a client is being counted, engine calls otherwise."""
        if self.counter is not None:
            submitted, completed, pubs = (
                self.counter.submitted,
                self.counter.completed,
                self.counter.pubs,
            )
        else:
            submitted = completed = self._engine_jobs
            pubs = 0
        return replace(
            snapshot,
            submitted=submitted,
            completed=completed,
            pubs=pubs,
            elapsed=self._elapsed(),
        )

    def _elapsed(self) -> float:
        return time.perf_counter() - self._started if self._started else 0.0

    def _notify(self) -> None:
        if self.listener is not None:
            self.listener(self.snapshot)


ValueOf = Callable[[str, Any], float | None]


def instrument(
    engine: Any, tracker: Tracker, value_of: ValueOf | None = None, run_kind: str = "objective"
) -> Any:
    """Wrap an engine's job-submitting calls to honour Stop and report each job.

    ``run``, ``run_gradient`` and ``batch_run`` are each one platform job. A call made
    from inside another is part of the outer job and is not counted again.
    """
    depth = 0

    def wrap(name: str, kind: str) -> None:
        inner = getattr(engine, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            nonlocal depth
            if depth:
                return inner(*args, **kwargs)
            tracker.check_stop()
            depth += 1
            try:
                result = inner(*args, **kwargs)
            finally:
                depth -= 1
            tracker.record_job(kind, value_of(kind, result) if value_of else None)
            return result

        setattr(engine, name, wrapped)

    wrap("run", run_kind)
    wrap("run_gradient", "gradient")
    wrap("batch_run", "batch")
    return engine
