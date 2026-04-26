"""
Sweep bounded-delay staleness and compare either:
  - test accuracy vs training iteration (default), or
  - global iteration throughput (iter/s) vs staleness (via bench timing).

Optionally include SEQUENTIAL_BSP and/or asynchronous references (throughput: horizontal lines; use --no-bsp-baseline / --no-async-baseline to skip).

From repo root:  python src/accuracy_delay_sweep.py
                 python src/accuracy_delay_sweep.py --metric throughput --delays 1,2,5,10
                 python src/accuracy_delay_sweep.py --metric throughput --steps-per-round 10
From src/:        python accuracy_delay_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bench_runtime import run_async_trial, run_one_trial  # noqa: E402
from config import (  # noqa: E402
    LEARNING_RATE,
    NUM_ITERATIONS,
    NUM_REPLICAS,
    NUM_SERVERS,
    NUM_WEIGHTS,
    NUM_WORKERS,
    TIMED_ITERS,
    WARMUP_ITERS,
    SyncMode,
)

_DEFAULT_EVAL_EVERY = 10
from main import run_training  # noqa: E402
from load_mnist import load_mnist_data  # noqa: E402


def parse_delays(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _mean_throughput_its(iter_times: list[float]) -> tuple[float, float]:
    """Return (throughput in global it/s, mean wall time per iteration in s)."""
    if not iter_times:
        return float("nan"), float("nan")
    m = statistics.mean(iter_times)
    thr = 1.0 / m if m > 0 else float("nan")
    return thr, m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bounded-delay staleness: test accuracy over time or iteration throughput."
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "throughput"],
        default="accuracy",
        help="accuracy: test accuracy vs iteration; throughput: mean global it/s vs staleness (default: accuracy).",
    )
    parser.add_argument(
        "--delays",
        type=str,
        default="1,2,5,10,20",
        help="Comma-separated staleness values to sweep (default: 1,2,5,10,20).",
    )
    parser.add_argument(
        "--out-figure",
        type=str,
        default=None,
        help="Output PNG path (defaults: plots/accuracy_delay_sweep/accuracy_vs_staleness.png or .../throughput_vs_staleness.png).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Output JSON path (default: same base name as --out-figure with .json).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path (default: same base name as --out-figure with .csv).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=f"Training iterations (default: {NUM_ITERATIONS} from config).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=_DEFAULT_EVAL_EVERY,
        help="Evaluate test accuracy every N iterations (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy random seed before each run (sharding shuffle; default: 0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of workers (default: {NUM_WORKERS}).",
    )
    parser.add_argument(
        "--no-bsp-baseline",
        action="store_true",
        help="Do not run SEQUENTIAL_BSP reference (accuracy or throughput).",
    )
    parser.add_argument(
        "--no-async-baseline",
        action="store_true",
        help="Do not run asynchronous reference (throughput only; one async step per worker per round).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter run: 40 iters (accuracy) or 1+5 timing samples (throughput).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=f"Timing warmup iterations discarded (throughput mode; default: {WARMUP_ITERS}).",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=None,
        help=f"Timed iterations for throughput mean (default: {TIMED_ITERS}).",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=NUM_REPLICAS,
        help=f"Hash-ring virtual nodes per server (default: {NUM_REPLICAS}).",
    )
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=1,
        help="Throughput only: for bounded delay, how many local steps per worker per timed "
        "round (default: 1). B>1 measures larger chunks (merges more tracker RPCs per round). "
        "Throughput 'it/s' is B-step rounds per second, not per single step.",
    )
    args = parser.parse_args()

    delays = parse_delays(args.delays)
    num_iterations: int | None = args.iterations
    eval_every = args.eval_every
    if args.quick and args.metric == "accuracy":
        if num_iterations is None:
            num_iterations = 40
        if args.eval_every == _DEFAULT_EVAL_EVERY:
            eval_every = 5

    warmup = args.warmup if args.warmup is not None else WARMUP_ITERS
    timed_iters = args.timed_iters if args.timed_iters is not None else TIMED_ITERS
    if args.quick and args.metric == "throughput":
        warmup = min(1, warmup)
        timed_iters = min(5, timed_iters)

    repo_root = os.path.dirname(_SRC)
    default_name = (
        "throughput_vs_staleness.png"
        if args.metric == "throughput"
        else "accuracy_vs_staleness.png"
    )
    out_fig = args.out_figure or os.path.join(
        repo_root, "plots", "accuracy_delay_sweep", default_name
    )
    base, _ = os.path.splitext(out_fig)
    out_json = args.out_json or f"{base}.json"
    out_csv = args.out_csv or f"{base}.csv"
    out_dir = os.path.dirname(os.path.abspath(out_fig))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    series: list[dict] = []
    flat_rows: list[dict] = []
    bsp_ref_throughput: dict | None = None
    async_ref_throughput: dict | None = None
    meta: dict

    if args.metric == "throughput":
        X_train, y_train, _, _ = load_mnist_data()
        bench_total_iters = warmup + timed_iters

        spr = max(1, int(args.steps_per_round))
        for s in delays:
            np.random.seed(args.seed)
            print(
                f"\n=== Throughput, bounded delay staleness={s} "
                f"(B={spr} local steps / worker / round) ==="
            )
            times = run_one_trial(
                num_workers=args.num_workers,
                num_servers=NUM_SERVERS,
                num_weights=NUM_WEIGHTS,
                num_iterations=bench_total_iters,
                num_replicas=args.num_replicas,
                learning_rate=LEARNING_RATE,
                X_train=X_train,
                y_train=y_train,
                warmup_iters=warmup,
                sync_mode=SyncMode.BOUNDED_DELAY,
                bounded_delay_staleness=s,
                steps_per_bound_round=spr,
            )
            thr, mean_s = _mean_throughput_its(times)
            n = len(times)
            print(
                f"  timed samples={n}  mean {mean_s*1000:.2f} ms/round  "
                f"throughput = {thr:.2f} B-step rounds/s"
            )
            label = f"s={s}"
            series.append(
                {
                    "name": label,
                    "mode": "bounded_delay",
                    "staleness": s,
                    "throughput_its": thr,
                    "mean_iter_s": mean_s,
                    "num_timed_samples": n,
                }
            )
            flat_rows.append(
                {
                    "run_label": label,
                    "mode": "bounded_delay",
                    "staleness": s,
                    "mean_iter_s": mean_s,
                    "throughput_its": thr,
                }
            )

        if not args.no_bsp_baseline:
            np.random.seed(args.seed)
            print("\n=== Throughput, SEQUENTIAL_BSP (reference) ===")
            times = run_one_trial(
                num_workers=args.num_workers,
                num_servers=NUM_SERVERS,
                num_weights=NUM_WEIGHTS,
                num_iterations=bench_total_iters,
                num_replicas=args.num_replicas,
                learning_rate=LEARNING_RATE,
                X_train=X_train,
                y_train=y_train,
                warmup_iters=warmup,
                sync_mode=SyncMode.SEQUENTIAL_BSP,
            )
            thr, mean_s = _mean_throughput_its(times)
            n = len(times)
            print(
                f"  timed samples={n}  mean {mean_s*1000:.2f} ms/iter  "
                f"throughput = {thr:.2f} global it/s"
            )
            bsp_ref_throughput = {
                "name": "sequential BSP",
                "mode": "sequential_bsp",
                "throughput_its": thr,
                "mean_iter_s": mean_s,
                "num_timed_samples": n,
            }
            flat_rows.append(
                {
                    "run_label": "sequential BSP",
                    "mode": "sequential_bsp",
                    "staleness": "",
                    "mean_iter_s": mean_s,
                    "throughput_its": thr,
                }
            )

        if not args.no_async_baseline:
            np.random.seed(args.seed)
            print("\n=== Throughput, ASYNCHRONOUS (reference) ===")
            times = run_async_trial(
                num_workers=args.num_workers,
                num_servers=NUM_SERVERS,
                num_weights=NUM_WEIGHTS,
                num_iterations=bench_total_iters,
                num_replicas=args.num_replicas,
                learning_rate=LEARNING_RATE,
                X_train=X_train,
                y_train=y_train,
                warmup_iters=warmup,
            )
            thr, mean_s = _mean_throughput_its(times)
            n = len(times)
            print(
                f"  timed samples={n}  mean {mean_s*1000:.2f} ms/round  "
                f"throughput = {thr:.2f} global rounds/s"
            )
            async_ref_throughput = {
                "name": "asynchronous",
                "mode": "asynchronous",
                "throughput_its": thr,
                "mean_iter_s": mean_s,
                "num_timed_samples": n,
            }
            flat_rows.append(
                {
                    "run_label": "asynchronous",
                    "mode": "asynchronous",
                    "staleness": "",
                    "mean_iter_s": mean_s,
                    "throughput_its": thr,
                }
            )

        meta = {
            "metric": "throughput",
            "num_workers": args.num_workers,
            "num_servers": NUM_SERVERS,
            "num_replicas": args.num_replicas,
            "num_weights": NUM_WEIGHTS,
            "learning_rate": LEARNING_RATE,
            "warmup_iters": warmup,
            "timed_iters": timed_iters,
            "bench_total_iters": bench_total_iters,
            "seed": args.seed,
            "steps_per_bound_round": spr,
            "throughput_unit": "B_step_rounds_per_s",
        }
        if bsp_ref_throughput is not None:
            meta["bsp_reference"] = bsp_ref_throughput
        if async_ref_throughput is not None:
            meta["async_reference"] = async_ref_throughput

    else:
        for s in delays:
            np.random.seed(args.seed)
            print(f"\n=== Bounded delay, staleness={s} ===")
            history = run_training(
                args.num_workers,
                NUM_WEIGHTS,
                LEARNING_RATE,
                sync_mode=SyncMode.BOUNDED_DELAY,
                bounded_delay_staleness=s,
                num_iterations=num_iterations,
                eval_every=eval_every,
                random_seed=args.seed,
            )
            label = f"s={s}"
            ser = {
                "name": label,
                "mode": "bounded_delay",
                "staleness": s,
                "points": history,
            }
            series.append(ser)
            for row in history:
                flat_rows.append(
                    {
                        "run_label": label,
                        "mode": "bounded_delay",
                        "staleness": s,
                        "iteration": row["iteration"],
                        "accuracy": row["accuracy"],
                    }
                )

        if not args.no_bsp_baseline:
            np.random.seed(args.seed)
            print("\n=== SEQUENTIAL_BSP (reference) ===")
            history = run_training(
                args.num_workers,
                NUM_WEIGHTS,
                LEARNING_RATE,
                sync_mode=SyncMode.SEQUENTIAL_BSP,
                num_iterations=num_iterations,
                eval_every=eval_every,
                random_seed=args.seed,
            )
            ser = {
                "name": "sequential BSP",
                "mode": "sequential_bsp",
                "staleness": None,
                "points": history,
            }
            series.append(ser)
            for row in history:
                flat_rows.append(
                    {
                        "run_label": "sequential BSP",
                        "mode": "sequential_bsp",
                        "staleness": "",
                        "iteration": row["iteration"],
                        "accuracy": row["accuracy"],
                    }
                )

        resolved_iters = num_iterations if num_iterations is not None else NUM_ITERATIONS
        meta = {
            "metric": "accuracy",
            "num_workers": args.num_workers,
            "num_servers": NUM_SERVERS,
            "num_replicas": args.num_replicas,
            "num_weights": NUM_WEIGHTS,
            "learning_rate": LEARNING_RATE,
            "num_iterations": resolved_iters,
            "eval_every": eval_every,
            "seed": args.seed,
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "series": series}, f, indent=2)

    if args.metric == "throughput":
        fieldnames = [
            "run_label",
            "mode",
            "staleness",
            "mean_iter_s",
            "throughput_its",
        ]
    else:
        fieldnames = [
            "run_label",
            "mode",
            "staleness",
            "iteration",
            "accuracy",
        ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    if args.metric == "throughput":
        bd = [s for s in series if s.get("mode") == "bounded_delay"]
        xs = [s["staleness"] for s in bd]
        ys = [s["throughput_its"] for s in bd]
        ax.plot(xs, ys, "o-", markersize=6, label="bounded delay")
        if bsp_ref_throughput is not None:
            bsp_t = bsp_ref_throughput["throughput_its"]
            ax.axhline(
                bsp_t,
                color="C1",
                linestyle="--",
                label=bsp_ref_throughput["name"],
            )
        if async_ref_throughput is not None:
            a_t = async_ref_throughput["throughput_its"]
            ax.axhline(
                a_t,
                color="C2",
                linestyle=":",
                label=async_ref_throughput["name"],
            )
        ax.set_xlabel("Staleness s")
        spr = max(1, int(args.steps_per_round))
        if spr == 1:
            ax.set_ylabel("Throughput (1 local step / worker / round; s⁻¹)")
        else:
            ax.set_ylabel(
                f"Throughput (B={spr} local steps / worker / round; s⁻¹)"
            )
        ax.set_title(
            "Training throughput vs staleness (ref: BSP & async = 1 step/round; "
            f"bounded delay uses B from --steps-per-round, default 1)"
        )
    else:
        for ser in series:
            line_style = "--" if ser.get("mode") == "sequential_bsp" else "-"
            pts = ser["points"]
            pxs = [p["iteration"] for p in pts]
            pys = [p["accuracy"] for p in pts]
            ax.plot(
                pxs,
                pys,
                linestyle=line_style,
                marker="o",
                markersize=3,
                label=ser["name"],
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Test accuracy vs iteration (bounded delay staleness sweep)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    print(f"\nWrote {out_fig}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
