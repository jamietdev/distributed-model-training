from collections import defaultdict

import ray
import numpy as np

from cluster import build_cluster, teardown_cluster
from config import (
    BOUNDED_DELAY_STALENESS,
    LEARNING_RATE,
    NUM_ITERATIONS,
    NUM_REPLICAS,
    NUM_SERVERS,
    NUM_WEIGHTS,
    NUM_WORKERS,
    SYNC_MODE,
    SyncMode,
)
from load_mnist import load_mnist_data

ray.shutdown()
ray.init()


def gather_full_weights(servers, hash_ring, num_weights):
    full_weights = np.zeros(num_weights)
    keys_by_server = defaultdict(list)
    for i in range(num_weights):
        server_id = hash_ring.get_server(i)
        keys_by_server[server_id].append(i)

    for server_id, indices in keys_by_server.items():
        weights_ref = servers[server_id].pull_weights.remote(indices)
        weights_dict: dict[int, float] = ray.get(weights_ref)
        for idx, val in weights_dict.items():
            full_weights[idx] = val

    return full_weights


def evaluate_global_model(weights, X_test, y_test):
    logits = X_test @ weights
    preds = 1 / (1 + np.exp(-logits))
    preds = (preds >= 0.5).astype(np.float32)
    return np.mean(preds == y_test)


def run_training(num_workers, num_weights, learning_rate, sync_mode=SYNC_MODE):
    X_train, y_train, X_test, y_test = load_mnist_data()

    ring, servers, workers, progress_tracker = build_cluster(
        num_workers=num_workers,
        num_servers=NUM_SERVERS,
        num_weights=num_weights,
        num_replicas=NUM_REPLICAS,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=sync_mode,
        bounded_delay_staleness=BOUNDED_DELAY_STALENESS,
    )

    training_history = []
    eval_every = 10

    try:
        if sync_mode == SyncMode.ASYNCHRONOUS:
            weights = gather_full_weights(servers, ring, num_weights)
            acc = evaluate_global_model(weights, X_test, y_test)
            print(f"Before training (async): Accuracy = {acc:.4f}")

            remaining = NUM_ITERATIONS
            total_steps = 0
            while remaining > 0:
                chunk = min(eval_every, remaining)
                ray.get([w.train_loop_async.remote(chunk) for w in workers])
                total_steps += chunk
                remaining -= chunk
                weights = gather_full_weights(servers, ring, num_weights)
                acc = evaluate_global_model(weights, X_test, y_test)
                print(f"After {total_steps} steps per worker (async): Accuracy = {acc:.4f}")
                training_history.append(
                    {"steps_per_worker": total_steps, "accuracy": float(acc)}
                )

        else:
            for iteration in range(NUM_ITERATIONS):
                futures = [worker.run_iteration.remote(iteration) for worker in workers]
                ray.get(futures)
                if iteration % eval_every == 0:
                    weights = gather_full_weights(servers, ring, num_weights)
                    acc = evaluate_global_model(weights, X_test, y_test)
                    label = "BSP" if sync_mode == SyncMode.SEQUENTIAL_BSP else "bounded_delay"
                    print(f"Iteration {iteration} ({label}): Accuracy = {acc:.4f}")
                training_history.append({"iteration": iteration})

    finally:
        teardown_cluster(servers, workers, progress_tracker)

    return training_history


if __name__ == "__main__":
    run_training(NUM_WORKERS, NUM_WEIGHTS, LEARNING_RATE, sync_mode=SYNC_MODE)
