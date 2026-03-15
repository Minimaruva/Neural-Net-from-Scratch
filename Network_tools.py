import numpy as np

def init_network(layer_dims, alpha=1, beta=1, **legacy_kwargs):
    weights = []
    biases = []

    if isinstance(layer_dims, int):
        input_size = layer_dims
        hidden_size = legacy_kwargs.get("hidden_size")
        labels_size = legacy_kwargs.get("labels_size", 1)
        num_hidden_layers = legacy_kwargs.get("num_hidden_layers", 3)

        if hidden_size is None:
            hidden_size = input_size + 1

        layer_dims = [input_size] + [hidden_size] * num_hidden_layers + [labels_size]

    if not isinstance(layer_dims, (list, tuple)) or len(layer_dims) < 2:
        raise ValueError("Network must have at least input and output layers.")

    for layer_index in range(1, len(layer_dims)):
        prev_dim = layer_dims[layer_index - 1]
        curr_dim = layer_dims[layer_index]
        weights.append(np.random.randn(curr_dim, prev_dim) * alpha)
        biases.append(np.random.randn(curr_dim, 1) * beta)

    return weights, biases

def sigmoid(arr):
    return 1 / (1+np.exp(-1*arr))


def sigmoid_derivative(activation):
    return activation * (1 - activation)


def feed_forward(input_layer, weights, biases, **legacy_kwargs):
        cache = [input_layer]

        for layer_index in range(len(weights)):
                prev_layer = cache[layer_index]
                W = weights[layer_index]
                b = biases[layer_index]
                Z = W @ prev_layer + b
                A = sigmoid(Z)
                cache.append(A)

        return cache


def predict(input_layer, weights, biases):
    return feed_forward(input_layer, weights, biases)[-1]


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def train_network(
    input_layer,
    labels,
    layer_dims,
    epochs=200,
    learning_rate=0.01,
    alpha=1,
    beta=1,
    progress_callback=None,
    stop_flag=None,
):
    weights, biases = init_network(layer_dims, alpha=alpha, beta=beta)
    loss_history = []

    samples = input_layer.shape[1]

    for epoch in range(epochs):
        if stop_flag is not None and stop_flag():
            break
        cache = feed_forward(input_layer, weights, biases)
        y_pred = cache[-1]

        loss = compute_mse(labels, y_pred)
        loss_history.append(loss)

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, loss)

        dA = (2 / samples) * (y_pred - labels)

        for layer_index in reversed(range(len(weights))):
            A_curr = cache[layer_index + 1]
            A_prev = cache[layer_index]

            dZ = dA * sigmoid_derivative(A_curr)
            dW = dZ @ A_prev.T
            db = np.sum(dZ, axis=1, keepdims=True)

            weights[layer_index] -= learning_rate * dW
            biases[layer_index] -= learning_rate * db

            dA = weights[layer_index].T @ dZ

    return weights, biases, loss_history


