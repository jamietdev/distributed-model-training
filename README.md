# distributed-model-training

Implementing https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf

distributed-model-training/
│
├── server.py            # Parameter server node
├── worker.py            # Worker node
├── hashing.py           # Consistent hashing
├── model.py             # Logistic regression logic
├── data.py              # MNIST loading + batching
├── coordinator.py       # Orchestrates training loop
├── fault_tolerance.py   # Checkpoint + recovery
├── simulation.py        # Run experiments
├── config.py            # Hyperparameters
└── utils.py             # Helper functions