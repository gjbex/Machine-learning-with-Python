# PyTorch

PyTorch is a popular open-source machine learning library developed by
Facebook's AI Research lab. It provides a flexible and dynamic computational
graph, making it easy to build and train deep learning models. PyTorch is
widely used in both academia and industry for various applications, including
natural language processing, computer vision, and reinforcement learning.


## What is it?

1. `mnist_data_exploration.ipynb`: Jupyter notebook that explores the MNIST
   dataset using PyTorch. It covers dataset representation, image semantics,
   handwriting variation, and class distributions.
1. `mnist_mlp.ipynb`: Jupyter notebook that trains and compares multilayer
   perceptrons with and without dropout on the MNIST dataset. It also covers
   checkpointing, confusion matrices, and sensitivity to initialization.
1. `mnist_cross_validation.ipynb`: Jupyter notebook that introduces stratified
   cross-validation with a compact MNIST multilayer perceptron. It covers
   paired model comparison, fold-to-fold variation, leakage prevention, and
   one final test-set evaluation.
1. `mnist_cnn.ipynb`: Jupyter notebook that demonstrates how to train a simple
   neural network on the MNIST dataset using PyTorch. It includes code for
   loading the dataset, defining the model architecture, training the model,
   and evaluating its performance.
1. `tensors.ipynb`: Jupyter notebook that provides an introduction to PyTorch
   tensors. It covers the basics of tensor operations, including creation,
   indexing, slicing, and mathematical operations. This notebook serves as a
   foundation for understanding how to work with data in PyTorch.
1. `ddp`: directory containing code to illustrate the use of Distributed Data
   Parallel (DDP) in PyTorch.
1. [`activation-checkpointing/`](activation-checkpointing/): hands-on material
   measuring the activation-memory versus recomputation trade-off in a native
   PyTorch DDP workload.
1. [`memory-diagnostics/`](memory-diagnostics/): hands-on exercise using
   phase-by-phase CUDA measurements to distinguish activation, optimizer-state,
   and retained-graph memory pressure.
1. [`fsdp/`](fsdp/): batch-friendly material comparing DDP with PyTorch FSDP2
   using a synthetic transformer.
