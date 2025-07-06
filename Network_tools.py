import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

hidden_layers = 3  # number of layers(excluding input layer)
features = 2  # number of features
outputs = 1  # number of output nodes

def init_network(input_size, hidden_size=None, labels_size=1, num_hidden_layers=3, alpha=1, beta=1):
    weights = []
    biases = []

    if hidden_size is None:
        hidden_size = input_size + 1

    # Input to first hidden layer
    weights.append(np.random.randn(hidden_size, input_size) * alpha)
    biases.append(np.random.randn(hidden_size, 1) * beta)
    
    # Hidden layers
    for i in range(num_hidden_layers - 1):
        weights.append(np.random.randn(hidden_size, hidden_size) * alpha)
        biases.append(np.random.randn(hidden_size, 1) * beta)
    
    # Last hidden to output layer
    weights.append(np.random.randn(labels_size, hidden_size) * alpha)
    biases.append(np.random.randn(labels_size, 1) * beta)
    
    return weights, biases

def sigmoid(arr):
    return 1 / (1+np.exp(-1*arr))


def feed_forward(input_layer, num_hidden_layers, weights, biases):
  cache = [input_layer]
  
  for i in range(0,num_hidden_layers+1):
    prev_layer = cache[i]
    W = weights[i]
    b = biases[i]
    Z = W @ prev_layer + b
    A = sigmoid(Z)
    cache.append(A)

  return cache


