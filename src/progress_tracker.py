import threading

import ray


# max_concurrency: Ray's default is one in-flight call per actor; with bounded delay, every
# worker can be inside wait_until_can_advance at the same time. We allow many concurrent
# methods and serialize access to `progress` with a lock/condition in-process (same actor).
@ray.remote(max_concurrency=256)
class ProgressTracker:
    """
    Tracks per-worker completed steps; enforces max staleness vs slowest worker.

    Threading: this actor may run multiple tracker methods concurrently (different Ray RPC
    threads). Shared state is guarded by `self._cond`. `report_completed_and_wait_to_start_next`
    is used in multi-step `run_bounded_session` to reduce per-step RPCs.
    """

    def __init__(self, worker_ids: list[str], staleness_x: int):
        # x=0 would forbid any lead and deadlock before the first step; treat as at least 1.
        self.staleness_x = max(1, staleness_x)
        self.progress = {wid: 0 for wid in worker_ids}
        self._cond = threading.Condition()

    def _can_advance(self, worker_id: str) -> bool:
        min_p = min(self.progress.values())
        my_p = self.progress[worker_id]
        return my_p + 1 - min_p <= self.staleness_x

    def wait_until_can_advance(self, worker_id: str) -> None:
        """Block until this worker may complete one more step (bounded delay)."""
        with self._cond:
            while not self._can_advance(worker_id):
                self._cond.wait()

    def report_completed_step(self, worker_id: str) -> None:
        with self._cond:
            self.progress[worker_id] += 1
            self._cond.notify_all()

    def report_completed_and_wait_to_start_next(self, worker_id: str) -> None:
        """
        One atomic unit for the *middle* of a multi-step run_bounded_session: count this step
        as completed, then block until the same predicate as `wait_until_can_advance` allows
        the next step. Halves Ray RPCs vs separate report + wait.
        """
        with self._cond:
            self.progress[worker_id] += 1
            self._cond.notify_all()
            while not self._can_advance(worker_id):
                self._cond.wait()

    def get_min_progress(self) -> int:
        with self._cond:
            return min(self.progress.values())
