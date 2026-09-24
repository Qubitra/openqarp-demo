"""The job counter, the tracker's states and the engine instrumentation."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from qubitra.errors import InsufficientCreditsError, ProviderError

from openqarp_demo.live import (
    CREDITS_PER_JOB,
    JobCounter,
    Snapshot,
    Status,
    Stopped,
    Tracker,
    instrument,
)


class FakeJobs:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def run(self, *, pubs: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self.calls == self.fail_on:
            raise ProviderError("transport failure: The read operation timed out")
        return f"job-{self.calls}"


class FakeClient:
    def __init__(self, fail_on: int | None = None) -> None:
        self.jobs = FakeJobs(fail_on)


def test_counter_counts_jobs_and_pubs() -> None:
    counter = JobCounter()
    client: Any = counter.wrap(FakeClient())  # type: ignore[arg-type]
    assert client.jobs.run(pubs=[1, 2, 3], backend_id="b") == "job-1"
    client.jobs.run(pubs=[1], backend_id="b")
    assert (counter.submitted, counter.completed, counter.pubs) == (2, 2, 4)


def test_counter_counts_a_submission_whose_response_is_lost() -> None:
    counter = JobCounter()
    client: Any = counter.wrap(FakeClient(fail_on=2))  # type: ignore[arg-type]
    client.jobs.run(pubs=[1], backend_id="b")
    with pytest.raises(ProviderError):
        client.jobs.run(pubs=[1], backend_id="b")
    assert (counter.submitted, counter.completed) == (2, 1)


class FakeEngine:
    def __init__(self) -> None:
        self.runs = 0

    def run(self, params: Any = None) -> list[float]:
        self.runs += 1
        return [-float(self.runs)]

    def run_gradient(self, params: Any = None, method: str = "default") -> list[float]:
        # A gradient built from the engine's own run is still one job.
        self.run(params)
        return [0.0]

    def batch_run(self, param_sets: Any) -> list[list[float]]:
        return [[0.0]]


def test_instrument_records_one_job_per_call_with_its_value() -> None:
    snapshots: list[Snapshot] = []
    tracker = Tracker(listener=snapshots.append)
    tracker.start()
    engine = instrument(
        FakeEngine(), tracker, lambda kind, r: -r[0] if kind == "objective" else None
    )
    engine.run({})
    engine.run_gradient({})
    engine.run({})
    points = tracker.snapshot.points
    assert [(p.job, p.kind, p.value) for p in points] == [
        (1, "objective", 1.0),
        (2, "gradient", None),
        (3, "objective", 3.0),
    ]
    assert tracker.snapshot.best == 3.0
    assert tracker.snapshot.submitted == 3
    assert tracker.snapshot.credits == 3 * CREDITS_PER_JOB
    assert snapshots[-1] == tracker.snapshot


def test_stop_is_honoured_before_the_next_job() -> None:
    tracker = Tracker()
    tracker.start()
    engine = instrument(FakeEngine(), tracker)
    engine.run({})
    tracker.stop()
    with pytest.raises(Stopped):
        engine.run({})
    assert engine.runs == 1


def test_a_transport_failure_is_a_dropped_connection_that_keeps_the_trace() -> None:
    counter = JobCounter()
    client: Any = counter.wrap(FakeClient(fail_on=3))  # type: ignore[arg-type]
    tracker = Tracker(counter=counter)
    tracker.start()

    class PlatformEngine(FakeEngine):
        def run(self, params: Any = None) -> list[float]:
            client.jobs.run(pubs=[1], backend_id="b")
            return super().run(params)

    engine = instrument(PlatformEngine(), tracker, lambda kind, r: -r[0])
    with pytest.raises(ProviderError) as raised:
        for _ in range(5):
            engine.run({})
    tracker.fail(raised.value)
    snapshot = tracker.snapshot
    assert snapshot.status is Status.DROPPED
    assert (snapshot.submitted, snapshot.completed) == (3, 2)
    assert len(snapshot.points) == 2
    assert "timed out" in snapshot.message
    # Nothing was resubmitted after the failure.
    assert client.jobs.calls == 3


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (Stopped(), Status.STOPPED),
        (ProviderError("PUB 0 failed on sim"), Status.FAILED),
        (InsufficientCreditsError("balance"), Status.FAILED),
        (RuntimeError("boom"), Status.FAILED),
    ],
)
def test_failures_map_to_a_status(exc: BaseException, status: Status) -> None:
    tracker = Tracker()
    tracker.start()
    tracker.fail(exc)
    assert tracker.snapshot.status is status


def test_listener_is_called_from_the_worker_thread() -> None:
    seen: list[Status] = []
    tracker = Tracker(listener=lambda s: seen.append(s.status))

    def work() -> None:
        tracker.start()
        tracker.record_job("objective", 1.0)
        tracker.finish("done")

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert seen == [Status.RUNNING, Status.RUNNING, Status.FINISHED]
    assert tracker.snapshot.outcome == "done"


def test_a_submission_redraws_before_its_job_returns() -> None:
    """The dashboard sees a job in flight: submitted is ahead of completed while it runs."""
    from types import SimpleNamespace

    from openqarp_demo.live import JobCounter, Status, Tracker

    seen: list[Snapshot] = []
    tracker = Tracker(listener=seen.append, counter=JobCounter())
    tracker.start()

    def run(*, pubs, **kwargs):
        in_flight = seen[-1]
        assert in_flight.status is Status.RUNNING
        assert (in_flight.submitted, in_flight.completed) == (1, 0)
        return "job"

    client = SimpleNamespace(jobs=SimpleNamespace(run=run))
    tracker.counter.wrap(client)  # type: ignore[union-attr, arg-type]
    client.jobs.run(pubs=[object()])
    assert tracker.counter.completed == 1  # type: ignore[union-attr]


def test_the_running_banner_shows_the_job_in_flight_and_a_progress_bar() -> None:
    from openqarp_demo import ui
    from openqarp_demo.live import Snapshot, Status

    html = ui.banner(Snapshot(status=Status.RUNNING, submitted=4, completed=3), "QAOA", 13)
    assert "Job 4 is running on the platform." in html
    assert "3 of ~13 jobs complete" in html
    assert 'class="oq-progress"' in html and 'class="oq-spinner"' in html
    idle = ui.banner(Snapshot(status=Status.RUNNING, submitted=3, completed=3), "QAOA", 13)
    assert "Preparing the next job." in idle
