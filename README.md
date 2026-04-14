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

To run training:
python3 main.py

TODOs:
- Currently we have sychronous training implemented, we should run it with different hyperparameters like learning rate changes (potentially try cross validation) and record the performance