# distributed-model-training

Implementing https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf.

More specifically, our project focuses on using the MNIST dataset along with a logistic regression model -- with the task of binary classification of whether an image of a number in the dataset is even or odd. We use 785 weights (784 pixels in each image + 1 bias) as we flatten the images to vectors in data pre-processing. The parameter servers exist to store certain subsets of weights, as determined by consistent hashing via a hash ring, while communicating with worker nodes, that compute the gradients. The parameter servers store the weights and aggregate the computed gradients across all the worker nodes, which are assigned different subsets of training data. The parameter servers then push the updated weights back to all of the worker nodes. This push and pull process then leads to our final weights, which can then be used for model predictions via running the main.py file. 

We then implemented a variety of experiments to test different aspects of our distributed system, such as different modes of synchronization (such as bounded delays vs. eventual vs. sequential).

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
