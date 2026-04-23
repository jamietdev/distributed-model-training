# distributed-model-training

Implementing https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf

```
distributed-model-training/
│
├── server.py            # Parameter server node
├── worker.py            # Worker node
├── hash_ring.py         # Consistent hashing
├── load_mnist.py        # MNIST loading + batching
├── config.py            # Hyperparameters
└── main.py              # Runs the script

Setup:
source venv/bin/activate
pip install -r requirements.txt
cd src

To run training:
python3 main.py

`BOUNDED_DELAY` is semi-asynchronous: workers are not global-barriered every step, and parameter servers apply each push with learning rate scaled by `1/num_workers` (vs. BSP, which averages all `N` gradients at once). Staleness `s` limits how far a worker can run ahead of the slowest (`ProgressTracker`).

TODOs:
- For pulls and pushes, in each iteration, each worker is sending gradients to each server. We could have the worker send all gradients in one rpc, and the server splits gradients internally. This could be another test to try 