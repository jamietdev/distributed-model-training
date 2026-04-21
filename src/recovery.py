import ray

from server import read_checkpoint_file


def reshard_after_failure(
    dead_server_id,
    ring,
    servers,
    workers,
):
    # read dead server's last checkpoint
    checkpoint = read_checkpoint_file(dead_server_id)
    if checkpoint is None:
        raise RuntimeError(
            f"No checkpoint found for dead server {dead_server_id}. "
            f"Cannot reshard. Check that CHECKPOINT_EVERY is set and at "
            f"least one checkpoint fired before the failure."
        )
    orphaned_weights = checkpoint["weights"]
    checkpoint_iter = checkpoint["iteration"]

    ring.remove_server(dead_server_id)

    # determining new owners of the dropped weights
    owner_map: dict[str, dict[int, float]] = {}
    for idx, val in orphaned_weights.items():
        new_owner = ring.get_server(idx)
        owner_map.setdefault(new_owner, {})[idx] = val

    # get each survivor to absorb its new weights
    absorb_refs = [] # references to ray actors
    absorb_targets = []
    for owner_id, weight_dict in owner_map.items():
        absorb_refs.append(servers[owner_id].absorb_weights.remote(weight_dict))
        absorb_targets.append(owner_id)
    new_sizes = ray.get(absorb_refs)

    # fix the weight map, remove the handle for the ray actor to the dead server
    del servers[dead_server_id]

    ray.get([w.refresh_weight_map.remote(ring) for w in workers])
    ray.get([w.remove_server_handle.remote(dead_server_id) for w in workers])

    return {
        "dead_server": dead_server_id,
        "checkpoint_iteration": checkpoint_iter,
        "orphaned_weight_count": len(orphaned_weights),
        "absorbed_by": {
            target: size for target, size in zip(absorb_targets, new_sizes)
        },
    }