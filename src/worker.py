from collections import defaultdict
import ray
import numpy as np

@ray.remote
class Worker:
    def __init__(
            self, 
            worker_id, 
            hash_ring,
            num_weights,
            learning_rate,
            X_train_batch, # this worker's shard of training data, a subset of the 784 rows
            y_train_batch,
            current_iteration,
            servers
            ):
        self.worker_id = worker_id
        self.hash_ring = hash_ring
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.X_train_batch = X_train_batch
        self.y_train_batch = y_train_batch
        self.current_iteration = current_iteration
        self.servers = servers
        self.local_weights = np.zeros(self.num_weights, dtype=np.float32)    
        self.weight_to_server_map = hash_ring.build_weight_map(num_weights)



    def run_iteration(
            self,
            iteration_num: int,
    ):
        self.current_iteration = iteration_num
        # pull weights from servers
        self.pull_weights()
        
        # sample batch
        X_batch = self.X_train_batch
        y_batch = self.y_train_batch
    
        # compute 
        gradients = self.compute_gradients(self.local_weights, X_batch, y_batch)

        # push like route each gradient to its owning server
        self.push_gradients(gradients)

        
    def compute_gradients(self, weights, X_batch, y_batch):
        # Forward pass 
        logits = X_batch @ weights
        preds = 1 / (1 + np.exp(-logits))

        # Error 
        error = preds - y_batch

        # Gradient
        gradients = X_batch.T @ error / len(X_batch)

        # Convert to dict for servers
        gradients_dict = {}
        for i in range(self.num_weights):
            gradients_dict[i] = gradients[i]

        return gradients_dict 
    
    def pull_weights(self):
        """
        Pulls weights from each server and stores them in local_weights
        
        First identify which servers own which weights by
        using the weight_to_server_map. 
        Then, it will pull the weights from
        each server using the pull_weights method of the ParameterServer class.
        Finally, it will store the pulled weights in local_weights.
        """
        servers_and_their_weights = defaultdict(list)
        for weightIdx, server_id in self.weight_to_server_map.items():
            servers_and_their_weights[server_id].append(weightIdx)
        
        # pull weights from servers
        for server_id, weight_indices in servers_and_their_weights.items():
            weights_ref = self.servers[server_id].pull_weights.remote(weight_indices, self.current_iteration)
            weights_dict: dict[int, float] = ray.get(weights_ref)

            # Step 3: store into local_weights
            for idx, val in weights_dict.items():
                self.local_weights[idx] = val

    
    # tells server, here are gradients, please update your weights
    def push_gradients(self, gradients):
        # group gradients by server
        grads_by_server = {}
        for idx, grad in gradients.items():
            server_id = self.hash_ring.get_server(idx)
            if server_id not in grads_by_server: 
                grads_by_server[server_id] = {}
            grads_by_server[server_id][idx] = grad

        # send to servers
        for server_id, grad_dict in grads_by_server.items():
            self.servers[server_id].push_gradients.remote(grad_dict, self.worker_id, self.current_iteration)
