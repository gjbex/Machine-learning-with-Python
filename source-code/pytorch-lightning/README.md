# Pythorch Ligntning

PyTorch Lightning is a lightweight wrapper for PyTorch that helps to organize
code and abstracts away the training loop, making it easier to write clean and
maintainable code. It provides a high-level interface for training neural
networks, allowing you to focus on the model architecture and the training
logic, rather than the boilerplate code for training and validation.


## What is it?

1. `mnist.ipynb`: A Jupyter notebook that demonstrates how to use PyTorch
   Lightning to train a simple neural network on the MNIST dataset. It includes
   code for data loading, model definition, training, and evaluation.
2. [`ddp/`](ddp/): A batch-friendly ResNet-50 benchmark for comparing one and
   multiple GPUs with PyTorch Lightning's DDP strategy.
3. [`fsdp/`](fsdp/): A batch-friendly synthetic-transformer benchmark for
   comparing Lightning DDP with FSDP2 memory sharding.
