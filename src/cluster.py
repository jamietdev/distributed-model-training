import numpy as np
import ray

from hash_ring import HashRing
from load_mnist import shard_data
from server import ParameterServer
from worker import Worker


def build_cluster(
    num_workers,
    num_servers,
    num_weights,
    num_replicas,
    learning_rate,
    X_train,
    y_train,
):
    shards = shard_data(X_train, y_train, num_workers)

    ring = HashRing(num_weights, num_replicas)
    server_ids = [f"server_{i}" for i in range(num_servers)]
    for sid in server_ids:
        ring.add_server(sid)

    servers = {}
    for sid in server_ids:
        owned = ring.get_weightIdxs_for_specific_server(sid)
        wvals = {k: float(np.random.randn() * 0.01) for k in owned}
        servers[sid] = ParameterServer.remote(
            server_id=sid,
            weight_indices=owned,
            num_weights=num_weights,
            learning_rate=learning_rate,
            weightVals=wvals,
            current_iteration=0,
            num_expected_workers=num_workers,
        )

    workers = []
    for i in range(num_workers):
        w = Worker.remote(
            worker_id=f"worker_{i}",
            hash_ring=ring,
            num_weights=num_weights,
            learning_rate=learning_rate,
            current_iteration=0,
            X_train_batch=shards[i][0],
            y_train_batch=shards[i][1],
            servers=servers,
        )
        workers.append(w)

    return ring, servers, workers


def teardown_cluster(servers, workers):
    for w in workers:
        ray.kill(w)
    for s in servers.values():
        ray.kill(s)