"""認識器の学習ジョブ（`runtime/detector_job.py`）を同時に始めたときの契約テスト（#89）。"""

from __future__ import annotations

import threading
import time

from app.runtime.detector_job import DetectorTrainingJob, parse_request

CALLERS = 8


class _Hooks:
    """start() が問い合わせるフックだけを持つ。実用モードの問い合わせで少し待ち、判定の隙間を広げる。"""

    def practical_mode(self) -> bool:
        time.sleep(0.02)
        return False

    def map_loading(self) -> bool:
        return False

    def autotune_running(self) -> bool:
        return False


def _job_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "detector-train"]


def test_concurrent_starts_run_one_job() -> None:
    request, why = parse_request({"mode": "full"})
    assert request is not None, why
    job = DetectorTrainingJob(_Hooks())  # type: ignore[arg-type]
    release = threading.Event()
    job._run = lambda _request: release.wait(10.0)  # type: ignore[method-assign]

    before = len(_job_threads())
    barrier = threading.Barrier(CALLERS)
    results: list[str | None] = [None] * CALLERS

    def call(i: int) -> None:
        barrier.wait()
        results[i] = job.start(request)

    callers = [threading.Thread(target=call, args=(i,)) for i in range(CALLERS)]
    try:
        for t in callers:
            t.start()
        for t in callers:
            t.join(10.0)
        started = len(_job_threads()) - before
        assert results.count("") == 1, results
        assert all("すでに学習を実行中" in (r or "") for r in results if r != ""), results
        assert started == 1
        assert job.running
    finally:
        release.set()
        job.stop(timeout=5.0)
    assert not job.running


def test_start_after_finish_runs_again() -> None:
    request, why = parse_request({"mode": "full"})
    assert request is not None, why
    job = DetectorTrainingJob(_Hooks())  # type: ignore[arg-type]
    job._run = lambda _request: None  # type: ignore[method-assign]
    assert job.start(request) == ""
    job.stop(timeout=5.0)
    assert not job.running
    assert job.start(request) == ""
    job.stop(timeout=5.0)
