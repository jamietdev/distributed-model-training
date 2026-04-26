"""
Sweep workers, servers, and hash-ring virtual nodes (replicas) for throughput/latency
using `bench_runtime.run_one_trial`.

Bounded delay: one line per value in `--delays` (default 1,2,5,10) plus sequential BSP
and asynchronous (3 lines for those, same y-axis, x = scale).

  python src/plot_iteration_trial_scaling.py
  python src/plot_iteration_trial_scaling.py --delays 1,2,5,20
  python src/plot_iteration_trial_scaling.py --delays 5   # single bounded-delay line (like old 3-line)
  python src/plot_iteration_trial_scaling.py --quick

Default output: plots/plot_iteration_trial_scaling/ (overridable with --output-dir).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ray

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bench_runtime import run_one_trial
from config import (
    BOUNDED_DELAY_STALENESS,
    LEARNING_RATE,
    NUM_WEIGHTS,
    SEED,
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

BSP_LABEL = "sequential BSP"
ASYNC_LABEL = "asynchronous"


def parse_delays(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def mean_throughput(times: list[float]) -> tuple[float, float]:
    if not times:
        return float("nan"), float("nan")
    m = statistics.mean(times)
    thr = 1.0 / m if m > 0 else float("nan")
    return m, thr


def _empty_series() -> dict[str, list]:
    return {"x": [], "latency_ms": [], "throughput": []}


def _collect_axis_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xs: list[int],
    delays: list[int],
    timed_iters: int,
    warmup_iters: int,
    trial_params: Callable[[int], dict],
    log_key: str,
    seed: int,
) -> dict[str, dict[str, list]]:
    """
    One series per bounded-delay staleness in `delays`, then BSP, then async
    (same as original 3-line plot, but with len(delays) lines for BD instead of 1).
    """
    sample_idx = 0
    out: dict[str, dict[str, list]] = {}
    for s in delays:
        out[f"bounded delay s={s}"] = _empty_series()
    out[BSP_LABEL] = _empty_series()
    out[ASYNC_LABEL] = _empty_series()

    for x in xs:
        p = trial_params(x)
        for s in delays:
            np.random.seed(seed + sample_idx)
            sample_idx += 1
            times = run_one_trial(
                num_weights=NUM_WEIGHTS,
                num_iterations=timed_iters + warmup_iters,
                learning_rate=LEARNING_RATE,
                X_train=X_train,
                y_train=y_train,
                sync_mode=SyncMode.BOUNDED_DELAY,
                bounded_delay_staleness=s,
                warmup_iters=warmup_iters,
                **p,
            )
            m, thr = mean_throughput(times)
            k = f"bounded delay s={s}"
            out[k]["x"].append(x)
            out[k]["latency_ms"].append(m * 1000.0)
            out[k]["throughput"].append(thr)
            print(
                f"  {log_key}={x}  bounded_delay s={s}  "
                f"mean_latency_ms={m*1000:.2f}  throughput_it_s={thr:.2f}"
            )

        np.random.seed(seed + sample_idx)
        sample_idx += 1
        times = run_one_trial(
            num_weights=NUM_WEIGHTS,
            num_iterations=timed_iters + warmup_iters,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            y_train=y_train,
            sync_mode=SyncMode.SEQUENTIAL_BSP,
            warmup_iters=warmup_iters,
            **p,
        )
        m, thr = mean_throughput(times)
        out[BSP_LABEL]["x"].append(x)
        out[BSP_LABEL]["latency_ms"].append(m * 1000.0)
        out[BSP_LABEL]["throughput"].append(thr)
        print(
            f"  {log_key}={x}  {BSP_LABEL}  "
            f"mean_latency_ms={m*1000:.2f}  throughput_it_s={thr:.2f}"
        )

        np.random.seed(seed + sample_idx)
        sample_idx += 1
        times = run_one_trial(
            num_weights=NUM_WEIGHTS,
            num_iterations=timed_iters + warmup_iters,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            y_train=y_train,
            sync_mode=SyncMode.ASYNCHRONOUS,
            warmup_iters=warmup_iters,
            **p,
        )
        m, thr = mean_throughput(times)
        out[ASYNC_LABEL]["x"].append(x)
        out[ASYNC_LABEL]["latency_ms"].append(m * 1000.0)
        out[ASYNC_LABEL]["throughput"].append(thr)
        print(
            f"  {log_key}={x}  {ASYNC_LABEL}  "
            f"mean_latency_ms={m*1000:.2f}  throughput_it_s={thr:.2f}"
        )

    return out


def _plot_series(
    series: dict[str, dict[str, list]],
    xlabel: str,
    ykey: str,
    ylabel: str,
    title: str,
    out_path: str,
    delays: list[int],
    xscale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = mpl.colormaps["viridis"]
    n_bd = max(1, len(delays))
    for j, s in enumerate(delays):
        k = f"bounded delay s={s}"
        d = series[k]
        c = cmap(0.2 + 0.7 * j / max(1, n_bd - 1) if n_bd > 1 else 0.5)
        ax.plot(
            d["x"],
            d[ykey],
            label=k,
            linestyle="-",
            color=c,
            marker="s",
            markersize=5,
        )
    bsp = series[BSP_LABEL]
    ax.plot(
        bsp["x"],
        bsp[ykey],
        label=BSP_LABEL,
        linestyle="--",
        color="C0",
        marker="o",
    )
    a = series[ASYNC_LABEL]
    ax.plot(
        a["x"],
        a[ykey],
        label=ASYNC_LABEL,
        linestyle=":",
        color="C1",
        marker="^",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scaling_pngs(
    by_workers: dict[str, dict[str, list]],
    by_servers: dict[str, dict[str, list]],
    by_replicas: dict[str, dict[str, list]],
    out_dir: str,
    file_prefix: str,
    delays: list[int],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    p = file_prefix
    _plot_series(
        by_workers,
        "Number of workers",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs workers (servers={FIXED_SERVERS})",
        os.path.join(out_dir, f"{p}latency_vs_workers.png"),
        delays,
    )
    _plot_series(
        by_workers,
        "Number of workers",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs workers (servers={FIXED_SERVERS})",
        os.path.join(out_dir, f"{p}throughput_vs_workers.png"),
        delays,
    )
    _plot_series(
        by_servers,
        "Number of parameter servers",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs servers (workers={FIXED_WORKERS})",
        os.path.join(out_dir, f"{p}latency_vs_servers.png"),
        delays,
    )
    _plot_series(
        by_servers,
        "Number of parameter servers",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs servers (workers={FIXED_WORKERS})",
        os.path.join(out_dir, f"{p}throughput_vs_servers.png"),
        delays,
    )
    _plot_series(
        by_replicas,
        "Virtual nodes per server (replicas)",
        "latency_ms",
        "Mean iteration latency (ms)",
        f"Latency vs replicas (workers={FIXED_REPLICA_WORKERS}, "
        f"servers={FIXED_REPLICA_SERVERS})",
        os.path.join(out_dir, f"{p}latency_vs_replicas.png"),
        delays,
        xscale="log",
    )
    _plot_series(
        by_replicas,
        "Virtual nodes per server (replicas)",
        "throughput",
        "Throughput (iterations / s)",
        f"Throughput vs replicas (workers={FIXED_REPLICA_WORKERS}, "
        f"servers={FIXED_REPLICA_SERVERS})",
        os.path.join(out_dir, f"{p}throughput_vs_replicas.png"),
        delays,
        xscale="log",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling benchmark: one bounded-delay line per --delays value, plus BSP & async."
    )
    parser.add_argument(
        "--delays",
        type=str,
        default="1,2,5,10",
        help=f"Comma-separated bounded-delay staleness s (one line per value). "
        f"Use e.g. `{BOUNDED_DELAY_STALENESS}` for a single line matching old config. "
        f"Default: 1,2,5,10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base RNG seed for trials.",
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
    delays = parse_delays(args.delays)
    if not delays:
        raise SystemExit("At least one value required for --delays")

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
        print(f"\n=== worker scaling (bounded delay s in {delays}, BSP, async) ===")
        by_workers = _collect_axis_sweep(
            X_train,
            y_train,
            worker_sweep,
            delays,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": n,
                "num_servers": FIXED_SERVERS,
                "num_replicas": NUM_REPLICAS,
            },
            "workers",
            args.seed,
        )
        print("\n=== server scaling ===")
        by_servers = _collect_axis_sweep(
            X_train,
            y_train,
            server_sweep,
            delays,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": FIXED_WORKERS,
                "num_servers": n,
                "num_replicas": NUM_REPLICAS,
            },
            "servers",
            args.seed,
        )
        print("\n=== replica (virtual node) scaling ===")
        by_replicas = _collect_axis_sweep(
            X_train,
            y_train,
            replica_sweep,
            delays,
            timed_iters,
            warmup_iters,
            lambda n: {
                "num_workers": FIXED_REPLICA_WORKERS,
                "num_servers": FIXED_REPLICA_SERVERS,
                "num_replicas": n,
            },
            "replicas",
            args.seed,
        )
        repo_root = os.path.dirname(_SRC)
        out_dir = args.output_dir or os.path.join(
            repo_root, "plots", "plot_iteration_trial_scaling"
        )
        prefix = args.file_prefix
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        plot_scaling_pngs(by_workers, by_servers, by_replicas, out_dir, prefix, delays)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
