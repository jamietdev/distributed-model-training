# owns a weight shard
# receives gradient updates, applies them, serve current values on request

import time

import ray

@ray.remote
class ParameterServer:
    def __init__(
            self, 
            server_id, 
            weight_indices,
            num_weights,
            learning_rate,
            weightVals,
            current_iteration,
            num_expected_workers):
        self.server_id = server_id
        self.weight_indices = weight_indices
        self.num_weights = num_weights 
        self.learning_rate = learning_rate
        self.weightVals = weightVals
        self.current_iteration = current_iteration
        self.workers_pushed_this_iter = 0
        self.gradient_store = {k: [] for k in weight_indices} # weight index -> gradient values from workers
        self.num_expected_workers = num_expected_workers
        self.workers_seen = set()
    
    def pull_weights(self, indices, expected_iteration=None) -> dict[int, float]: # returns dict of weight index -> weight value for requested indices
        # workers call this at the start of each iteration
        
        # wait until server catches up
        if expected_iteration is not None:
            while self.current_iteration < expected_iteration:
                time.sleep(0.001)
        return {i: self.weightVals[i] for i in indices}

    def push_gradients(self, gradient_dict: dict[int, float], worker_id, iteration) -> None: 
        """
        A worker calls this. "here are my gradient updates for the weights this server owns"
        
        If the iteration number provided does not match the current iteration, the call is ignored.
        
        Otherwise, the gradient updates are stored in the gradient store. Once all workers have pushed their gradients for this iteration, the server will average the gradients and apply the update to the weights.
        """
        if iteration != self.current_iteration:
            return
        for idx, grad in gradient_dict.items():
            assert idx in self.gradient_store, f"Unexpected weight index {idx}"
            self.gradient_store[idx].append(grad)
        
        self.workers_seen.add(worker_id)
        if len(self.workers_seen) == self.num_expected_workers:
            self.update_weights()
    
   
    def update_weights(self):
        """
        Called after all workers have pushed their gradients for this iteration.
        Averages the gradients and applies the update to the weights.
        Resets the gradient store and iteration number.
        """
        self.workers_seen = set()
        for weightIndex in self.weight_indices:
            grads = self.gradient_store[weightIndex]
            if len(grads) == 0:
                continue
            average_gradient = sum(grads) / len(grads)
            self.weightVals[weightIndex] -= self.learning_rate * average_gradient
        
        # reset for next iteration
        self.gradient_store = {weightIdx: [] for weightIdx in self.weight_indices}
        self.workers_pushed_this_iter = 0
        self.current_iteration += 1

    def get_iteration(self) -> int:
        return self.current_iteration


    