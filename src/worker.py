from collections import defaultdict

import ray
import numpy as np

from config import SyncMode


@ray.remote
class Worker:
    def __init__(
        self,
        worker_id,
        hash_ring,
        num_weights,
        learning_rate,
        X_train_batch,
        y_train_batch,
        current_iteration,
        servers,
        sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
        progress_tracker=None,
    ):
        self.worker_id = worker_id
        self.hash_ring = hash_ring
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.X_train_batch = X_train_batch
        self.y_train_batch = y_train_batch
        self.current_iteration = current_iteration
        self.servers = servers
        self.sync_mode = sync_mode
        self.progress_tracker = progress_tracker
        self.local_weights = np.zeros(self.num_weights, dtype=np.float32)
        self.weight_to_server_map = hash_ring.build_weight_map()

    def run_iteration(self, iteration_num: int):
        print(f"[{self.worker_id}] iter={self.current_iteration}")
        self.current_iteration = iteration_num
        if self.progress_tracker is not None:
            ray.get(self.progress_tracker.wait_until_can_advance.remote(self.worker_id))

        self.pull_weights(wait_for_iteration=True)

        X_batch = self.X_train_batch
        y_batch = self.y_train_batch
        gradients = self.compute_gradients(self.local_weights, X_batch, y_batch)
        self.push_gradients(gradients)

        if self.progress_tracker is not None:
            ray.get(self.progress_tracker.report_completed_step.remote(self.worker_id))

    def train_loop_async(self, num_steps: int):
        """Independent steps; servers apply gradients immediately (no BSP barrier)."""
        for _ in range(num_steps):
            self.pull_weights(wait_for_iteration=False)
            X_batch = self.X_train_batch
            y_batch = self.y_train_batch
            gradients = self.compute_gradients(self.local_weights, X_batch, y_batch)
            self.push_gradients(gradients)

    def compute_gradients(self, weights, X_batch, y_batch):
        # Forward pass 
        logits = X_batch @ weights
        preds = 1 / (1 + np.exp(-logits))

        # Error 
        error = preds - y_batch

        # Gradient
        gradients = X_batch.T @ error / len(X_batch)

        # Convert to dict for servers
        gradients_dict = {}
        for i in range(self.num_weights):
            gradients_dict[i] = gradients[i]
        return gradients_dict


    def pull_weights(self, wait_for_iteration: bool = True):
        """
        Pulls weights from each server and stores them in local_weights
        """

        servers_and_their_weights = defaultdict(list)
        for weight_index, server_id in self.weight_to_server_map.items():
            servers_and_their_weights[server_id].append(weight_index)

        expected = (
            self.current_iteration if wait_for_iteration else None
        )
        refs = []
        server_ids = []
        weight_lists = []

        for server_id, weight_indices in servers_and_their_weights.items():
            refs.append(
                self.servers[server_id].pull_weights.remote(weight_indices, expected)
            )
            server_ids.append(server_id)
            weight_lists.append(weight_indices)

        results = ray.get(refs)

        for weights_dict in results:
            for idx, val in weights_dict.items():
                self.local_weights[idx] = val
            
    # tells server, here are gradients, please update your weights
    def push_gradients(self, gradients):
        # group gradients by server
        grads_by_server = {}
        for idx, grad in gradients.items():
            server_id = self.weight_to_server_map[idx]
            if server_id not in grads_by_server:
                grads_by_server[server_id] = {}
            grads_by_server[server_id][idx] = grad

        # send to servers
        refs = []
        for server_id, grad_dict in grads_by_server.items():
            iteration = None if self.sync_mode == SyncMode.ASYNCHRONOUS else self.current_iteration
            refs.append(
                self.servers[server_id].push_gradients.remote(
                    grad_dict, self.worker_id, iteration
                )
            )

        if self.sync_mode != SyncMode.ASYNCHRONOUS:
            ray.get(refs)
    
    def reconfigure(self, ring, servers):
        self.hash_ring = ring
        self.servers = dict(servers)
        # rebuild weight map
        self.weight_to_server_map = ring.build_weight_map()

    def set_iteration(self, iteration):
        self.current_iteration = iteration