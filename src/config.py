from enum import Enum


class SyncMode(str, Enum):
    """Distributed training synchronization strategy."""

    SEQUENTIAL_BSP = "sequential_bsp"
    ASYNCHRONOUS = "asynchronous"
    BOUNDED_DELAY = "bounded_delay"


NUM_SERVERS = 3
NUM_WORKERS = 10
NUM_WEIGHTS = 785  # 784 input features + 1 bias
NUM_ITERATIONS = 50
LEARNING_RATE = 0.2
NUM_REPLICAS = 2

SYNC_MODE = SyncMode.SEQUENTIAL_BSP
# BOUNDED_DELAY: workers run local run_bounded_session with ProgressTracker (at most this many
# steps ahead of the slowest); each gradient applies immediately with lr/num_workers per shard
# (stale reads; no all-worker barrier each step), unlike SEQUENTIAL_BSP. ProgressTracker uses a
# Ray actor with max_concurrency plus an in-actor lock/condition (see progress_tracker.py).
BOUNDED_DELAY_STALENESS = 5

# for runtime testing
WARMUP_ITERS = 5  # discarded; first few iters are dominated by JIT/RPC setup
TIMED_ITERS = 30  # samples actually used for stats
ITER_TIMEOUT_S = 10.0

# for fault tolerance
RECOVERY_MODE = "chain" # or "chain"
CHAIN_REPLICAS = 2
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 50

SEED = 67
