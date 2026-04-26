"""
Sweep workers, servers, and hash-ring virtual nodes (replicas) for each SyncMode
using `bench_runtime.run_one_trial`; write latency and throughput vs each
dimension (mean iteration time from the timed loop).

  python src/plot_iteration_trial_scaling.py
  python src/plot_iteration_trial_scaling.py --quick

Default output: plots/plot_iteration_trial_scaling/ (overridable with --output-dir).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections.abc import Callable

import matplotlib.pyplot as plt
import ray

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bench_runtime import run_one_trial
from config import (
    LEARNING_RATE,
    NUM_WEIGHTS,
    TIMED_ITERS,
    WARMUP_ITERS,
    SyncMode,
)
from load_mnist import load_mnist_data

FIXED_SERVERS = 2
FIXED_WORKERS = 6
NUM_REPLICAS = 2

WORKER_SWEEP_FULL = [1, 2, 4, 6, 8]
SERVER_SWEEP_FULL = [1, 2, 4, 8]
FIXED_REPLICA_WORKERS = 6
FIXED_REPLICA_SERVERS = 4
REPLICA_SWEEP_FULL = [1, 2, 10, 50, 200]

SYNC_LABELS = {
    SyncMode.SEQUENTIAL_BSP: "sequential BSP",
    SyncMode.BOUNDED_DELAY: "bounded delay",
    SyncMode.ASYNCHRONOUS: "asynchronous",
}

SYNC_STYLES = {
    SyncMode.SEQUENTIAL_BSP: dict(color="C0", marker="o"),
    SyncMode.BOUNDED_DELAY: dict(color="C1", marker="s"),
    SyncMode.ASYNCHRONOUS: dict(color="C2", marker="^"),
}


def mean_throughput(times: list[float]) -> tuple[float, float]:
    if not times:
        return float("nan"), float("nan")
    m = statistics.mean(times)
    thr = 1.0 / m if m > 0 else float("nan")
    return m, thr


def _collect_axis_sweep(
    X_train,
    y_train,
    xs: list[int],
    timed_iters: int,
    warmup_iters: int,
    trial_params: Callable[[int], dict],
    log_key: str,
) -> dict[SyncMode, dict[str, list]]:
    out: dict[SyncMode, dict[str, list]] = {}
    for mode in SyncMode:
        latencies = []
        throughputs = []
        for x in xs:
            times = run_one_trial(
                num_weights=NUM_WEIGHTS,
                num_iterations=timed_iters + warmup_iters,
                learning_rate=LEARNING_RATE,
                X_train=X_train,
                y_train=y_train,
                sync_mode=mode,
                warmup_iters=warmup_iters,
                **trial_params(x),
            )
            m, thr = mean_throughput(times)
            latencies.append(m * 1000.0)
            throughputs.append(thr)
            print(
                f"  {log_key}={x} sync={mode.value} "
                f"mean_latency_ms={m*1000:.2f} throughput_it_s={thr:.2f}"
            )
        out[mode] = {"x": list(xs), "latency_ms": latencies, "throughput": throughputs}
    return out


def _plot_single(
    data_map: dict[SyncMode, dict[str, list]],
    xlabel: str,
    ykey: str,
    ylabel: str,
    title: str,
    out_path: str,
    xscale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in SyncMode:
        d = data_map[mode]
        sty = SYNC_STYLES[mode]
        ax.plot(
            d["x"],
            d[ykey],
            label=SYNC_LABELS[mode],
            linestyle="-",
            **sty,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scaling_pngs(
    by_workers: dict[SyncMode, dict[str, list]],
    by_servers: dict[SyncMode, dict[str, list]],
    by_replicas: dict[SyncMode, dict[str, list]],
    out_dir: str,
    file_prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    p = file_prefix
    _plot_single(
        by_workers,
        "Number of workers",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs workers (servers={FIXED_SERVERS})",
        os.path.join(out_dir, f"{p}latency_vs_workers.png"),
    )
    _plot_single(
        by_workers,
        "Number of workers",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs workers (servers={FIXED_SERVERS})",
        os.path.join(out_dir, f"{p}throughput_vs_workers.png"),
    )
    _plot_single(
        by_servers,
        "Number of parameter servers",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs servers (workers={FIXED_WORKERS})",
        os.path.join(out_dir, f"{p}latency_vs_servers.png"),
    )
    _plot_single(
        by_servers,
        "Number of parameter servers",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs servers (workers={FIXED_WORKERS})",
        os.path.join(out_dir, f"{p}throughput_vs_servers.png"),
    )
    _plot_single(
        by_replicas,
        "Virtual nodes per server (replicas)",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs replicas (workers={FIXED_REPLICA_WORKERS}, "
        f"servers={FIXED_REPLICA_SERVERS})",
        os.path.join(out_dir, f"{p}latency_vs_replicas.png"),
        xscale="log",
    )
    _plot_single(
        by_replicas,
        "Virtual nodes per server (replicas)",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs replicas (workers={FIXED_REPLICA_WORKERS}, "
        f"servers={FIXED_REPLICA_SERVERS})",
        os.path.join(out_dir, f"{p}throughput_vs_replicas.png"),
        xscale="log",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteration-timing benchmark: sweep scale and plot latency/throughput."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller sweeps and fewer timed iterations (smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs (default: plots/plot_iteration_trial_scaling).",
    )
    parser.add_argument(
        "--file-prefix",
        default="",
        help="Optional prefix for filenames (e.g. 'run1' -> run1_latency_vs_workers.png, ...).",
    )
    args = parser.parse_args()

    if args.quick:
        worker_sweep = [2, 4, 6]
        server_sweep = [1, 2, 4]
        replica_sweep = [1, 2, 10]
        timed_iters = 5
        warmup_iters = 1
    else:
        worker_sweep = WORKER_SWEEP_FULL
        server_sweep = SERVER_SWEEP_FULL
        replica_sweep = REPLICA_SWEEP_FULL
        timed_iters = TIMED_ITERS
        warmup_iters = WARMUP_ITERS

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        X_train, y_train, _, _ = load_mnist_data()
        print("\n=== Collecting worker scaling (all sync modes) ===")
        by_workers = _collect_axis_sweep(
            X_train,
            y_train,
            worker_sweep,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": n,
                "num_servers": FIXED_SERVERS,
                "num_replicas": NUM_REPLICAS,
            },
            "workers",
        )
        print("\n=== Collecting server scaling (all sync modes) ===")
        by_servers = _collect_axis_sweep(
            X_train,
            y_train,
            server_sweep,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": FIXED_WORKERS,
                "num_servers": n,
                "num_replicas": NUM_REPLICAS,
            },
            "servers",
        )
        print("\n=== Collecting replica (virtual node) scaling (all sync modes) ===")
        by_replicas = _collect_axis_sweep(
            X_train,
            y_train,
            replica_sweep,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": FIXED_REPLICA_WORKERS,
                "num_servers": FIXED_REPLICA_SERVERS,
                "num_replicas": n,
            },
            "replicas",
        )
        repo_root = os.path.dirname(_SRC)
        out_dir = args.output_dir or os.path.join(
            repo_root, "plots", "plot_iteration_trial_scaling"
        )
        prefix = args.file_prefix
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        plot_scaling_pngs(by_workers, by_servers, by_replicas, out_dir, prefix)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
