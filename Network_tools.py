import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

hidden_layers = 3  # number of layers(excluding input layer)
features = 2  # number of features
outputs = 1  # number of output nodes

def init_network(input_size=features, hidden_size=features+1, output_size=outputs, num_hidden_layers=hidden_layers, alpha=1, beta=1):
    weights = []
    biases = []
    
    alpha = 1  # scaling factor for weights
    beta = 1   # scaling factor for biases

    # Input to first hidden layer
    weights.append(np.random.randn(hidden_size, input_size) * alpha)
    biases.append(np.random.randn(hidden_size, 1) * beta)
    
    # Hidden layers
    for i in range(num_hidden_layers - 1):
        weights.append(np.random.randn(hidden_size, hidden_size) * alpha)
        biases.append(np.random.randn(hidden_size, 1) * beta)
    
    # Last hidden to output layer
    weights.append(np.random.randn(output_size, hidden_size) * alpha)
    biases.append(np.random.randn(output_size, 1) * beta)
    
    return weights, biases


