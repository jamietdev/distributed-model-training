import time
import statistics

import numpy as np
import ray

from cluster import build_cluster, teardown_cluster
from hash_ring import HashRing
from load_mnist import load_mnist_data
from config import LEARNING_RATE, NUM_WEIGHTS, WARMUP_ITERS, TIMED_ITERS, ITER_TIMEOUT_S, NUM_WORKERS, NUM_SERVERS, NUM_REPLICAS

def summarize(samples, label):
    s = sorted(samples)
    n = len(s)
    mean = statistics.mean(s)
    std = statistics.stdev(s) if n > 1 else 0.0
    p50 = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    throughput = 1.0 / mean if mean > 0 else float("inf")
    print(
        f"  {label:30s} n={n:3d}  "
        f"mean={mean*1000:7.2f}ms  std={std*1000:6.2f}ms  "
        f"p50={p50*1000:7.2f}ms  p95={p95*1000:7.2f}ms  "
        f"throughput={throughput:6.1f} it/s"
    )
    return {"mean": mean, "std": std, "p50": p50, "p95": p95}


def run_one_trial(
    num_workers,
    num_servers,
    num_weights,
    num_iterations,
    num_replicas,
    learning_rate,
    X_train,
    y_train,
    warmup_iters=WARMUP_ITERS,
):
    _, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=num_weights,
        num_replicas=num_replicas,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
    )

    # timed loop
    iter_times = []
    try:
        for it in range(num_iterations):
            t0 = time.perf_counter()
            ray.get(
                [w.run_iteration.remote(it) for w in workers],
                timeout=ITER_TIMEOUT_S,
            )
            iter_times.append(time.perf_counter() - t0)
    except ray.exceptions.GetTimeoutError:
        print(
            f"  !! iteration {len(iter_times)} timed out after "
            f"{ITER_TIMEOUT_S}s — likely the pull_weights busy-wait deadlock"
        )
    finally:
        teardown_cluster(servers, workers)

    return iter_times[warmup_iters:]


def bench_baseline(X_train, y_train):
    print("\n=== Baseline (workers=6, servers=2, replicas=2) ===")
    times = run_one_trial(
        num_workers=6, num_servers=2,
        num_weights=NUM_WEIGHTS, num_iterations=TIMED_ITERS + WARMUP_ITERS,
        num_replicas=2, learning_rate=LEARNING_RATE,
        X_train=X_train, y_train=y_train,
    )
    summarize(times, "per-iter")
    print(f"  total wall-clock (timed): {sum(times):.2f}s")


def bench_scaling_workers(X_train, y_train):
    """Hold servers fixed, vary workers. Expect: latency rises as each
    server has to wait for more pushes before update_weights() fires."""
    print("\n=== Scaling: num_workers (servers=2, replicas=2) ===")
    for nw in [1, 2, 4, 6, 8]:
        times = run_one_trial(
            num_workers=nw, num_servers=2,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=2, learning_rate=LEARNING_RATE,
            X_train=X_train, y_train=y_train,
        )
        summarize(times, f"workers={nw}")


def bench_scaling_servers(X_train, y_train):
    """Hold workers fixed, vary servers. With more servers, each worker
    fans out more RPCs per pull/push — so this often gets *worse*, not
    better, at this problem scale. Useful sanity check."""
    print("\n=== Scaling: num_servers (workers=6, replicas=2) ===")
    for ns in [1, 2, 4, 8]:
        times = run_one_trial(
            num_workers=NUM_WORKERS, num_servers=NUM_SERVERS,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=NUM_REPLICAS, learning_rate=LEARNING_RATE,
            X_train=X_train, y_train=y_train,
        )
        summarize(times, f"servers={ns}")


def bench_replicas(X_train, y_train):
    """Vary virtual nodes per server. Per-iter latency should be largely
    insensitive — the cost shows up at startup in build_weight_map and in
    add/remove operations, not the steady-state hot path."""
    print("\n=== Virtual-node count (workers=6, servers=4) ===")
    for nr in [1, 2, 10, 50, 200]:
        times = run_one_trial(
            num_workers=N, num_servers=4,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=nr, learning_rate=LEARNING_RATE,
            X_train=X_train, y_train=y_train,
        )
        summarize(times, f"replicas={nr}")


def bench_load_balance():
    """Not a runtime metric per se, but worth printing: how lopsided
    is the weight-to-server assignment at low replica counts?"""
    print("\n=== Load distribution by replica count (servers=4) ===")
    for nr in [1, 2, 10, 50, 200]:
        ring = HashRing(NUM_WEIGHTS, nr)
        for i in range(4):
            ring.add_server(f"server_{i}")
        by_server = ring.weightIdxs_for_all_servers(NUM_WEIGHTS)
        counts = sorted(len(v) for v in by_server.values())
        ideal = NUM_WEIGHTS / 4
        max_deviation = max(abs(c - ideal) for c in counts) / ideal
        print(
            f"  replicas={nr:3d}  shard sizes={counts}  "
            f"max_dev={max_deviation*100:5.1f}% from ideal ({ideal:.1f})"
        )


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    X_train, y_train, _, _ = load_mnist_data()

    bench_baseline(X_train, y_train)
    bench_load_balance()
    bench_scaling_workers(X_train, y_train)
    bench_scaling_servers(X_train, y_train)
    bench_replicas(X_train, y_train)

    ray.shutdown()