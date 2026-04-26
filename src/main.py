from collections import defaultdict
import random

import ray
import numpy as np
import os

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
    CHECKPOINT_DIR,
    SEED,
)
from load_mnist import load_mnist_data

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

def clear_checkpoints():
       if not os.path.isdir(CHECKPOINT_DIR):
           os.makedirs(CHECKPOINT_DIR, exist_ok=True)
           return
       for f in os.listdir(CHECKPOINT_DIR):
           os.remove(os.path.join(CHECKPOINT_DIR, f))


def run_training(
    num_workers,
    num_weights,
    learning_rate,
    sync_mode=SYNC_MODE,
    bounded_delay_staleness: int | None = None,
    num_iterations: int | None = None,
    eval_every: int = 10,
    random_seed: int | None = None,
):
    if bounded_delay_staleness is None:
        bounded_delay_staleness = BOUNDED_DELAY_STALENESS
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS
    rng_seed = SEED if random_seed is None else random_seed

    np.random.seed(rng_seed)
    random.seed(rng_seed)
    X_train, y_train, X_test, y_test = load_mnist_data()
    clear_checkpoints()
    # Reseed before sharding: shard_data() uses np.random; cluster uses random.Random(rng_seed)
    # for weight init, so re-seed here to match a fresh run's shard split.
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    ring, servers, workers, progress_tracker = build_cluster(
        num_workers=num_workers,
        num_servers=NUM_SERVERS,
        num_weights=num_weights,
        num_replicas=NUM_REPLICAS,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=sync_mode,
        bounded_delay_staleness=bounded_delay_staleness,
        weight_init_seed=rng_seed,
    )

    training_history = []
    try:
        w0 = gather_full_weights(servers, ring, num_weights)
        acc0 = evaluate_global_model(w0, X_test, y_test)
        training_history.append(
            {
                "iteration": -1,
                "steps_per_worker": 0,
                "before_training": True,
                "accuracy": float(acc0),
            }
        )
        if sync_mode == SyncMode.ASYNCHRONOUS:
            print(f"Before training (async): Accuracy = {acc0:.4f}")
        elif sync_mode == SyncMode.BOUNDED_DELAY:
            print(f"Before training (bounded_delay): Accuracy = {acc0:.4f}")
        else:
            print(f"Before training (sequential BSP): Accuracy = {acc0:.4f}")

        if sync_mode == SyncMode.ASYNCHRONOUS:
            remaining = num_iterations
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
                    {
                        "iteration": int(total_steps),
                        "steps_per_worker": int(total_steps),
                        "before_training": False,
                        "accuracy": float(acc),
                    }
                )

        elif sync_mode == SyncMode.BOUNDED_DELAY:
            remaining = num_iterations
            total_steps = 0
            while remaining > 0:
                chunk = min(eval_every, remaining)
                ray.get([w.run_bounded_session.remote(chunk) for w in workers])
                total_steps += chunk
                remaining -= chunk
                weights = gather_full_weights(servers, ring, num_weights)
                acc = evaluate_global_model(weights, X_test, y_test)
                print(
                    f"After {total_steps} steps per worker (bounded_delay): Accuracy = {acc:.4f}"
                )
                training_history.append(
                    {
                        "iteration": int(total_steps),
                        "steps_per_worker": int(total_steps),
                        "before_training": False,
                        "accuracy": float(acc),
                    }
                )

        else:
            for iteration in range(num_iterations):
                ray.get([worker.run_iteration.remote(iteration) for worker in workers])
                if iteration % eval_every == 0:
                    weights = gather_full_weights(servers, ring, num_weights)
                    acc = evaluate_global_model(weights, X_test, y_test)
                    print(
                        f"Iteration {iteration} (sequential BSP): Accuracy = {acc:.4f}"
                    )
                    training_history.append(
                        {
                            "iteration": int(iteration),
                            "before_training": False,
                            "accuracy": float(acc),
                        }
                    )

    finally:
        teardown_cluster(servers, workers, progress_tracker)

    return training_history


if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    run_training(NUM_WORKERS, NUM_WEIGHTS, LEARNING_RATE, sync_mode=SYNC_MODE)
