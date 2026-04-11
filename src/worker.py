class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
    
    def pull_weights(self, servers):
        TODO
    
    def compute_gradients(self, batch):
        TODO
    
    def push_gradients(self, gradients, servers):
        TODO