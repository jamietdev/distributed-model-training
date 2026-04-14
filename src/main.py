
from collections import defaultdict
import ray
import numpy as np
from config import LEARNING_RATE, NUM_SERVERS, NUM_WEIGHTS, NUM_WORKERS, NUM_ITERATIONS, NUM_REPLICAS
from hash_ring import HashRing
from load_mnist import load_mnist_data, shard_data
from server import ParameterServer
from worker import Worker

ray.shutdown() # to restart ray cleanly
ray.init()

def gather_full_weights(servers, hash_ring, num_weights):
    # reconstruct full weight vector
    full_weights = np.zeros(num_weights)

    keys_by_server = defaultdict(list)

    for i in range(num_weights):
        server_id = hash_ring.get_server(i)
        keys_by_server[server_id].append(i)

    # pull from each server
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


def run_training(num_workers, num_weights, learning_rate):
    # load dataset and shard data to workers
    X_train, y_train, X_test, y_test = load_mnist_data()
    shards = shard_data(X_train, y_train, num_workers)

    # build hash ring
    ring = HashRing(num_weights, NUM_REPLICAS)
    server_ids = [f"server_{i}" for i in range(NUM_SERVERS)]
    servers = {}
    for serv_id in server_ids:
        ring.add_server(serv_id)

    # start servers
    for serv_id in server_ids:
        owned_keys = ring.get_weightIdxs_for_specific_server(serv_id)
        weightVals = {k: np.random.randn() * 0.01 for k in owned_keys} # initialize weights to random values
        servers[serv_id] = ParameterServer.remote(
            server_id=serv_id,
            weight_indices=owned_keys,
            num_weights=num_weights,
            learning_rate=learning_rate,
            weightVals=weightVals,
            current_iteration=0,
            num_expected_workers=num_workers,
        )
        

    # start workers
    workers = []
    for i in range(num_workers):
        worker = Worker.remote(
            worker_id=f"worker_{i}",
            hash_ring=ring,
            num_weights=num_weights,
            learning_rate=learning_rate,
            current_iteration=0,
            X_train_batch=shards[i][0],
            y_train_batch=shards[i][1],
            servers=servers,   # 👈 VERY IMPORTANT
        )
        workers.append(worker)

    # training loop
    training_history = []
    losses = []
    for iteration in range(NUM_ITERATIONS):
        futures = [worker.run_iteration.remote(iteration) for worker in workers]
        ray.get(futures)
        if iteration % 10 == 0:
            weights = gather_full_weights(servers, ring, num_weights)
            acc = evaluate_global_model(weights, X_test, y_test)
            print(f"Iteration {iteration}: Accuracy = {acc:.4f}")
        training_history.append({
            "iteration": iteration,
        })
    
    return training_history

if __name__ == "__main__":
    run_training(NUM_WORKERS, NUM_WEIGHTS, LEARNING_RATE)