from enum import Enum


class SyncMode(str, Enum):
    """Distributed training synchronization strategy."""

    SEQUENTIAL_BSP = "sequential_bsp"
    ASYNCHRONOUS = "asynchronous"
    BOUNDED_DELAY = "bounded_delay"


NUM_SERVERS = 2
NUM_WORKERS = 6
NUM_WEIGHTS = 785  # 784 input features + 1 bias
NUM_ITERATIONS = 130
LEARNING_RATE = 0.2
NUM_REPLICAS = 2

SYNC_MODE = SyncMode.SEQUENTIAL_BSP
# BOUNDED_DELAY: workers run local run_bounded_session with ProgressTracker (at most this many
# steps ahead of the slowest); each gradient applies immediately with lr/num_workers per shard
# (stale reads; no all-worker barrier each step), unlike SEQUENTIAL_BSP.
BOUNDED_DELAY_STALENESS = 5

# for runtime testing
WARMUP_ITERS = 3  # discarded; first few iters are dominated by JIT/RPC setup
TIMED_ITERS = 30  # samples actually used for stats
ITER_TIMEOUT_S = 10.0
