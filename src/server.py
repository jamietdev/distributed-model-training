# owns a weight shard
# receives gradient updates, applies them, serve current values on request

import time
import json
import ray
import os

from config import SyncMode, CHECKPOINT_DIR, CHECKPOINT_EVERY

# dead server's actor is gone; need this to read its last checkpoint from disk to know updated values
# need this here b/c of ray flag
def read_checkpoint_file(server_id):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{server_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "iteration": data["iteration"],
        "weights": {int(k): v for k, v in data["weights"].items()},
    }

@ray.remote
class ParameterServer:
    def __init__(
        self,
        server_id,
        weight_indices,
        num_weights,
        learning_rate,
        weight_vals,
        current_iteration,
        num_expected_workers,
        sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
    ):
        self.server_id = server_id
        self.weight_indices = list(weight_indices)
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.weight_vals = dict(weight_vals)
        self.current_iteration = current_iteration
        self.gradient_store = {k: [] for k in self.weight_indices}
        self.num_expected_workers = num_expected_workers
        self.workers_seen = set()
        self.sync_mode = sync_mode
        self.async_updates = sync_mode == SyncMode.ASYNCHRONOUS

    def pull_weights(self, indices, expected_iteration=None) -> dict[int, float]:
        if not self.async_updates and expected_iteration is not None:
            while self.current_iteration < expected_iteration:
                time.sleep(0.001)
        return {i: self.weight_vals[i] for i in indices}

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
            self.weight_vals[idx] -= self.learning_rate * grad
        # self.current_iteration += 1

    def update_weights(self):
        self.workers_seen = set()
        for weight_index in self.weight_indices:
            grads = self.gradient_store[weight_index]
            if len(grads) == 0:
                continue
            average_gradient = sum(grads) / len(grads)
            self.weight_vals[weight_index] -= self.learning_rate * average_gradient

        self.gradient_store = {weight_index: [] for weight_index in self.weight_indices}
        self.current_iteration += 1

        if CHECKPOINT_EVERY > 0 and self.current_iteration % CHECKPOINT_EVERY == 0:
            self.add_checkpoint()
    
    # absorb the additional weights inherited from a failed server
    def absorb_weights(self, new_weight_dict: dict[int, float]) -> int:
        for idx, val in new_weight_dict.items():
            assert idx not in self.gradient_store, (
                f"{self.server_id} already owns weight {idx}; cannot absorb"
            )
            self.weight_indices.append(idx)
            self.weight_vals[idx] = val
            self.gradient_store[idx] = []
        return len(self.weight_indices)

    def get_iteration(self) -> int:
        return self.current_iteration

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def get_weight_indices(self) -> list:
        return list(self.weight_indices)

    def _checkpoint_path(self) -> str:
        return os.path.join(CHECKPOINT_DIR, f"checkpoint_{self.server_id}.json")

    def add_checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_data = {
            "iteration": int(self.current_iteration),
            "weights": {
                str(i): float(self.weight_vals[i]) for i in self.weight_indices
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
        self.weight_vals = loaded_weights
        self.current_iteration = data["iteration"]
        return self.current_iteration