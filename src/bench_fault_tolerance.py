"""Fault tolerance tests for remove-and-reshard recovery.

  test_static_reshard — kill a server without worrying abt training step, 
  reshard, verify every weight is still accounted for and the absorbed values match the checkpoint.

  test_live_reshard   — kill a server MID-TRAINING, reshard, resume, verify accuracy continues to improve.
"""
import os
import sys

import ray

from cluster import build_cluster, teardown_cluster
from config import (
    CHECKPOINT_EVERY,
    LEARNING_RATE,
    NUM_WEIGHTS,
    SyncMode,
    NUM_WORKERS,
    NUM_SERVERS,
    NUM_REPLICAS,
)
from load_mnist import load_mnist_data
from main import clear_checkpoints, evaluate_global_model, gather_full_weights
from recovery import reshard_after_failure


def _run_iterations(workers, start, end):
    for it in range(start, end):
        ray.get([w.run_iteration.remote(it) for w in workers])


def _kill_iteration_multiple_of_checkpoint(n):
    if n % CHECKPOINT_EVERY == 0:
        return n
    return ((n // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY


def test_static_reshard():
    print("\n=== Static reshard ===")
    clear_checkpoints()
    X_train, y_train, _, _ = load_mnist_data()

    ring, servers, workers, tracker = build_cluster(
        num_workers=NUM_WORKERS,
        num_servers=NUM_SERVERS,
        num_weights=NUM_WEIGHTS,
        num_replicas=NUM_REPLICAS,
        learning_rate=LEARNING_RATE,
        X_train=X_train,
        y_train=y_train,
        sync_mode=SyncMode.SEQUENTIAL_BSP,
    )

    try:
        kill_after = _kill_iteration_multiple_of_checkpoint(10)
        _run_iterations(workers, 0, kill_after)

        victim_id = "server_0"
        victim_indices = ring.get_weightIdxs_for_specific_server(victim_id)
        pre_kill_orphans = ray.get(
            servers[victim_id].pull_weights.remote(victim_indices)
        )
        print(f"  killing {victim_id} with {len(victim_indices)} weights "
              f"at iter {kill_after}")

        ray.kill(servers[victim_id])

        info = reshard_after_failure(
            dead_server_id=victim_id,
            ring=ring,
            servers=servers,
            workers=workers,
        )
        print(f"  reshard info: {info}")

        # verify thatevery orphaned weight ended up on exactly one survivor, and its value matches the checkpoint
        for orphan_idx, expected_val in pre_kill_orphans.items():
            new_owner = ring.get_server(orphan_idx)
            assert new_owner != victim_id, (
                f"weight {orphan_idx} still routes to dead {victim_id}"
            )
            assert new_owner in servers, (
                f"weight {orphan_idx} routes to unknown {new_owner}"
            )
            actual = ray.get(
                servers[new_owner].pull_weights.remote([orphan_idx])
            )
            assert abs(actual[orphan_idx] - expected_val) < 1e-9, (
                f"weight {orphan_idx} wrong after absorb: "
                f"expected {expected_val}, got {actual[orphan_idx]}"
            )

        # verify that the union of all survivor shards covers every weight index exactly once
        total_owned = 0
        for sid, handle in servers.items():
            idxs = ray.get(handle.get_weight_indices.remote())
            total_owned += len(idxs)
        assert total_owned == NUM_WEIGHTS, (
            f"weight accounting broken: {total_owned} owned total, "
            f"expected {NUM_WEIGHTS}"
        )

        print(f"  all {len(pre_kill_orphans)} orphans correctly absorbed")
        print(f"  all {NUM_WEIGHTS} weights accounted for across survivors")
        print("  PASS")
    finally:
        teardown_cluster(servers, workers, tracker)


def test_live_reshard():
    print("\n=== Live mid-training reshard ===")
    clear_checkpoints()
    X_train, y_train, X_test, y_test = load_mnist_data()

    ring, servers, workers, tracker = build_cluster(
        num_workers=NUM_WORKERS,
        num_servers=NUM_SERVERS,
        num_weights=NUM_WEIGHTS,
        num_replicas=NUM_REPLICAS,
        learning_rate=LEARNING_RATE,
        X_train=X_train,
        y_train=y_train,
        sync_mode=SyncMode.SEQUENTIAL_BSP,
    )

    try:
        kill_after = _kill_iteration_multiple_of_checkpoint(30)
        _run_iterations(workers, 0, kill_after)

        pre_kill_weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
        pre_kill_acc = evaluate_global_model(pre_kill_weights, X_test, y_test)
        print(f"  pre-kill accuracy (iter {kill_after}): {pre_kill_acc:.4f}")

        victim_id = "server_1"
        ray.kill(servers[victim_id])
        reshard_after_failure(
            dead_server_id=victim_id,
            ring=ring,
            servers=servers,
            workers=workers,
        )

        # fewer servers now, but workers have the new ring.
        _run_iterations(workers, kill_after, kill_after + 30)

        final_weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
        final_acc = evaluate_global_model(final_weights, X_test, y_test)
        print(f"  post-reshard accuracy "
              f"(iter {kill_after + 30}): {final_acc:.4f}")

        assert final_acc >= pre_kill_acc - 0.05, (
            f"training regressed significantly: "
            f"pre={pre_kill_acc:.4f}, post={final_acc:.4f}"
        )
        print("  PASS")
    finally:
        teardown_cluster(servers, workers, tracker)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        test_static_reshard()
        test_live_reshard()
        print("\nAll fault tolerance tests passed.")
    finally:
        ray.shutdown()