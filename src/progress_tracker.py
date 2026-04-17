import time

import ray


@ray.remote
class ProgressTracker:
    """Tracks per-worker completed steps; enforces max staleness vs slowest worker."""

    def __init__(self, worker_ids: list[str], staleness_x: int):
        # x=0 would forbid any lead and deadlock before the first step; treat as at least 1.
        self.staleness_x = max(1, staleness_x)
        self.progress = {wid: 0 for wid in worker_ids}

    def wait_until_can_advance(self, worker_id: str) -> None:
        """Block until this worker may complete one more step (bounded delay)."""
        while True:
            min_p = min(self.progress.values())
            my_p = self.progress[worker_id]
            if my_p + 1 - min_p <= self.staleness_x:
                return
            time.sleep(0.001)

    def report_completed_step(self, worker_id: str) -> None:
        self.progress[worker_id] += 1

    def get_min_progress(self) -> int:
        return min(self.progress.values())
