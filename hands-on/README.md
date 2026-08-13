# Hands-on notebooks

The core notebooks use PyTorch for the neural-network examples. Start Jupyter
Lab from this directory so that downloaded datasets and generated model files
remain below `hands-on`:

```bash
cd hands-on
jupyter lab
```

All notebooks come in at least two versions:
  
  * `lazy`: all code is ready to execute and contains no saved output; and
  * `complete`: the reference solution, including saved output, intended for
    viewing on GitHub.

For some notebooks, there is a `courageous` version as well, which means
that you will have to complete consequential parts of the code yourself.

1. 010_underfitting_overfitting:
    illustrates the concepts of underfitting and and overfitting using
    non-linear regression.
1. 020_mnist_data_exploration:
    explores the MNIST data set and PyTorch's `Dataset` representation.
1. 030_activation_function:
    visualization of the relevant activation functions.
1. 040_mnist_mlp:
    illustrates PyTorch data loaders, explicit training and evaluation loops,
    and construction of a classic multilayer perceptron to recognize
    handwritten digits.
1. 050_convolution:
    illustrates convolution as used in convolutional neural networks.
1. 060_mnist_cnn:
    trains and compares convolutional neural networks in PyTorch to recognize
    handwritten digits.

The `optional` directory contains the legacy Keras-based IMDB/RNN exercises.
They are retained as legacy supplementary material and are not part of the
PyTorch core path.
