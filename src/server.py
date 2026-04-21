# owns a weight shard
# receives gradient updates, applies them, serve current values on request

import time
import json
import ray
import os

from config import SyncMode, CHECKPOINT_DIR, CHECKPOINT_EVERY

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

    def pull_weights(self, indices, expected_iteration=None) -> dict[int, float]:
        if not self.async_updates and expected_iteration is not None:
            while self.current_iteration < expected_iteration:
                time.sleep(0.001)
        return {i: self.weightVals[i] for i in indices}

    def push_gradients(
        self, gradient_dict: dict[int, float], worker_id, iteration
    ) -> None:
        if self.async_updates:
            self._apply_gradients_immediate(gradient_dict)
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
        # self.current_iteration += 1

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

        if CHECKPOINT_EVERY > 0 and self.current_iteration % CHECKPOINT_EVERY == 0:
            self.add_checkpoint()

    def get_iteration(self) -> int:
        return self.current_iteration

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def _checkpoint_path(self) -> str:
        return os.path.join(CHECKPOINT_DIR, f"checkpoint_{self.server_id}.json")

    def add_checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_data = {
            "iteration": self.current_iteration,
            "weights": {
                str(i): self.weightVals[i] for i in self.weight_indices
            },
        }
        with open(self._checkpoint_path(), "w") as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
 
        loaded_weights = {int(k): v for k, v in data["weights"].items()}
 
        # Fail if ring assignment changed under us
        assert set(loaded_weights.keys()) == set(self.weight_indices), (
            f"Checkpoint indices don't match current shard assignment for "
            f"{self.server_id}. Checkpoint has "
            f"{len(loaded_weights)} indices, server owns "
            f"{len(self.weight_indices)}."
        )
        self.weightVals = loaded_weights
        self.current_iteration = data["iteration"]
        return self.current_iteration