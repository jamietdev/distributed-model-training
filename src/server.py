# owns a weight shard
# receives gradient updates, applies them, serve current values on request

import time

import ray

from config import SyncMode


@ray.remote
class ParameterServer:
    def __init__(
        self,
        server_id,
        weight_indices,
        num_weights,
        learning_rate,
        weightVals,
        current_iteration,
        num_expected_workers,
        sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
    ):
        self.server_id = server_id
        self.weight_indices = weight_indices
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.weightVals = weightVals
        self.current_iteration = current_iteration
        self.workers_pushed_this_iter = 0
        self.gradient_store = {k: [] for k in weight_indices}
        self.num_expected_workers = num_expected_workers
        self.workers_seen = set()
        self.sync_mode = sync_mode
        self.async_updates = sync_mode == SyncMode.ASYNCHRONOUS
        self.bounded_delay = sync_mode == SyncMode.BOUNDED_DELAY

    def pull_weights(self, indices, expected_iteration=None) -> dict[int, float]:
        # Stale / latest: async and bounded delay do not block until a global BSP clock.
        wait_bsp = (
            not self.async_updates
            and not self.bounded_delay
            and expected_iteration is not None
        )
        if wait_bsp:
            while self.current_iteration < expected_iteration:
                time.sleep(0.001)
        return {i: self.weightVals[i] for i in indices}

    def push_gradients(
        self, gradient_dict: dict[int, float], worker_id, iteration
    ) -> None:
        if self.async_updates:
            self._apply_gradients_immediate(gradient_dict)
            return

        if self.bounded_delay:
            # Middle ground: one partial SGD step per push, scaled 1/N vs BSP (average of N).
            for idx, grad in gradient_dict.items():
                assert idx in self.gradient_store, f"Unexpected weight index {idx}"
                self.weightVals[idx] -= (
                    (self.learning_rate / self.num_expected_workers) * grad
                )
            self.current_iteration += 1
            return

        if iteration != self.current_iteration:
            return
        for idx, grad in gradient_dict.items():
            assert idx in self.gradient_store, f"Unexpected weight index {idx}"
            self.gradient_store[idx].append(grad)

        self.workers_seen.add(worker_id)
        if len(self.workers_seen) == self.num_expected_workers:
            self.update_weights()

    def _apply_gradients_immediate(self, gradient_dict: dict[int, float]) -> None:
        for idx, grad in gradient_dict.items():
            assert idx in self.gradient_store, f"Unexpected weight index {idx}"
            self.weightVals[idx] -= self.learning_rate * grad
        self.current_iteration += 1

    def update_weights(self):
        self.workers_seen = set()
        for weightIndex in self.weight_indices:
            grads = self.gradient_store[weightIndex]
            if len(grads) == 0:
                continue
            average_gradient = sum(grads) / len(grads)
            self.weightVals[weightIndex] -= self.learning_rate * average_gradient

        self.gradient_store = {weightIdx: [] for weightIdx in self.weight_indices}
        self.workers_pushed_this_iter = 0
        self.current_iteration += 1

    def get_iteration(self) -> int:
        return self.current_iteration

    def add_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
