NUM_SERVERS = 2
NUM_WORKERS = 6
NUM_WEIGHTS = 785 # 784 input features + 1 bias
NUM_ITERATIONS = 130
LEARNING_RATE = 0.2 
NUM_REPLICAS = 2

# for runtime testing
WARMUP_ITERS = 3       # discarded; first few iters are dominated by JIT/RPC setup
TIMED_ITERS = 30       # samples actually used for stats
ITER_TIMEOUT_S = 10.0 
