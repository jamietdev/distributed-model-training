from __future__ import annotations

import time
import os
import sys
import matplotlib.pyplot as plt
import ray

# allow running from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cluster import build_cluster, teardown_cluster
from config import (
    LEARNING_RATE,
    NUM_WEIGHTS,
    SyncMode,
)
from load_mnist import load_mnist_data
from main import evaluate_global_model, gather_full_weights


WORKER_SWEEP = [1, 2, 4, 6, 8]
SERVER_SWEEP = [1, 2, 4, 8]
REPLICA_SWEEP = [1, 2, 10, 50, 200]

FIXED_WORKERS = 6
FIXED_SERVERS = 2
FIXED_REPLICA_WORKERS = 6
FIXED_REPLICA_SERVERS = 4

STEPS_PER_WORKER = 30

SYNC_LABELS = {
    SyncMode.SEQUENTIAL_BSP: "BSP",
    SyncMode.BOUNDED_DELAY: "bounded delay",
    SyncMode.ASYNCHRONOUS: "async",
}


def run_throughput_trial(
    num_workers,
    num_servers,
    num_replicas,
    X,
    y,
    mode,
):
    ring, servers, workers, tracker = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=NUM_WEIGHTS,
        num_replicas=num_replicas,
        learning_rate=LEARNING_RATE,
        X_train=X,
        y_train=y,
        sync_mode=mode,
    )

    start = time.perf_counter()

    if mode == SyncMode.ASYNCHRONOUS:
        ray.get([w.train_loop_async.remote(STEPS_PER_WORKER) for w in workers])
    else:
        for it in range(STEPS_PER_WORKER):
            ray.get([w.run_iteration.remote(it) for w in workers])

    elapsed = time.perf_counter() - start
    teardown_cluster(servers, workers, tracker)

    total_updates = num_workers * STEPS_PER_WORKER
    return total_updates / elapsed  # updates/sec


# accuracy vs time
def run_accuracy_curve(num_workers, num_servers, num_replicas, X, y, X_test, y_test, mode):
    ring, servers, workers, tracker = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=NUM_WEIGHTS,
        num_replicas=num_replicas,
        learning_rate=LEARNING_RATE,
        X_train=X,
        y_train=y,
        sync_mode=mode,
    )

    times = []
    accs = []

    start = time.perf_counter()
    total_steps = 0
    eval_every = 10

    try:
        if mode == SyncMode.ASYNCHRONOUS:
            weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
            acc = evaluate_global_model(weights, X_test, y_test)

            times.append(0)
            accs.append(acc)
            while total_steps < 100:
                ray.get([w.train_loop_async.remote(eval_every) for w in workers])
                total_steps += eval_every
                weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
                acc = evaluate_global_model(weights, X_test, y_test)

                times.append(time.perf_counter() - start)
                accs.append(acc)

        else:
            weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
            acc = evaluate_global_model(weights, X_test, y_test)

            times.append(0)
            accs.append(acc)
            for it in range(100):
                ray.get([w.run_iteration.remote(it) for w in workers])

                if it % eval_every == 0:
                    weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
                    acc = evaluate_global_model(weights, X_test, y_test)

                    times.append(time.perf_counter() - start)
                    accs.append(acc)

    finally:
        teardown_cluster(servers, workers, tracker)

    return times, accs



def plot_lines(x_values, results, xlabel, ylabel, title, filename, logx=False):
    plt.figure()

    for mode in SyncMode:
        plt.plot(
            x_values,
            results[mode],
            marker="o",
            label=SYNC_LABELS[mode],
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if logx:
        plt.xscale("log")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()



def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    X, y, X_test, y_test = load_mnist_data()

    # throughput vs workers
    results = {mode: [] for mode in SyncMode}
    for nw in WORKER_SWEEP:
        for mode in SyncMode:
            t = run_throughput_trial(
                nw, FIXED_SERVERS, 2, X, y, mode
            )
            results[mode].append(t)

    plot_lines(
        WORKER_SWEEP,
        results,
        "Workers",
        "Updates / sec",
        "Throughput vs Workers",
        "throughput_workers.png",
    )

    # throughput vs servers
    results = {mode: [] for mode in SyncMode}
    for ns in SERVER_SWEEP:
        for mode in SyncMode:
            t = run_throughput_trial(
                FIXED_WORKERS, ns, 2, X, y, mode
            )
            results[mode].append(t)

    plot_lines(
        SERVER_SWEEP,
        results,
        "Servers",
        "Updates / sec",
        "Throughput vs Servers",
        "throughput_servers.png",
    )

    # throughput vs replicas
    results = {mode: [] for mode in SyncMode}
    for nr in REPLICA_SWEEP:
        for mode in SyncMode:
            t = run_throughput_trial(
                FIXED_REPLICA_WORKERS,
                FIXED_REPLICA_SERVERS,
                nr,
                X,
                y,
                mode,
            )
            results[mode].append(t)

    plot_lines(
        REPLICA_SWEEP,
        results,
        "Replicas",
        "Updates / sec",
        "Throughput vs Replicas",
        "throughput_replicas.png",
        logx=True,
    )

    # accuracy vs time
    plt.figure()

    for mode in SyncMode:
        times, accs = run_accuracy_curve(
            FIXED_WORKERS,
            FIXED_SERVERS,
            2,
            X,
            y,
            X_test,
            y_test,
            mode,
        )
        plt.plot(times, accs, marker="o", label=SYNC_LABELS[mode])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Wall-Clock Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_time.png", dpi=150)
    plt.close()

    ray.shutdown()


if __name__ == "__main__":
    main()