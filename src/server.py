class ParameterServer:
    def __init__(self, server_id, weight_indices):
        self.server_id = server_id
        self.weight_indices = weight_indices
        self.weights = {}
    
    def pull_weights(self, indices):
        TODO

    def push_gradients(self, gradients):
        TODO
    
    def save_checkpoint(self):
        TODO
    