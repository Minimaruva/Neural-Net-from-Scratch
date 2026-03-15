# Overview
A simple Neural network with backpropagation from scratch in python using common packages like numpy and pandas

The neural network is generalised for any number of layers and inputs

Titanic dataset from https://www.kaggle.com/competitions/titanic/data
Idea inspired from this tutorial https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

The project contains the notebooks with neural network as well as demo of
neural network performance on titanic dataset

The basic data preparation functions are implemented too to give a way for general user
to train neural network on reasobaly usable datasets

The gui is planned to be done online

# Challenges
...

# Demo and results

# How to use
## GUI data prep
- Run `python gui.py` to open the data loader UI.
- Pick training and testing CSV files, or switch to single-file split mode.
- Choose the label column and preprocessing options.
- Optionally select columns to drop before preprocessing.
- Click "Process Data" to store `x_data` and `y_data` in memory.

## Network initialization
- Use the generalized API: `init_network([input_dim, hidden1, hidden2, output_dim])`.
- `feed_forward(input_layer, weights, biases)` now iterates over all layers automatically.

## Notebooks
- The existing notebooks are kept intact as test/reference networks.

