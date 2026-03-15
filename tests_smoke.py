import numpy as np
import pandas as pd

from data_prep import (
    encode_non_numeric,
    fill_missing_values,
    normalize_data_z_score,
    split_reshape_data,
)
from Network_tools import train_network, predict


def test_data_prep():
    df = pd.DataFrame(
        {
            "age": [22, None, 35, 28],
            "sex": ["male", "female", "female", "male"],
            "fare": [7.25, 71.83, None, 8.05],
            "label": [0, 1, 1, 0],
        }
    )

    encoded = encode_non_numeric(df.copy())
    assert np.issubdtype(encoded["sex"].dtype, np.number)

    filled = fill_missing_values(encoded.copy())
    assert not filled[["age", "fare"]].isna().any().any()

    normalized, mean, std = normalize_data_z_score(filled.copy(), labels=["label"])
    feature_cols = [col for col in normalized.columns if col != "label"]
    feature_mean = normalized[feature_cols].mean().abs().max()
    assert feature_mean < 1e-6
    assert all(std[feature_cols] > 0)

    x_data, y_data = split_reshape_data(normalized, ["label"])
    assert x_data.shape[1] == y_data.shape[1]
    assert x_data.shape[0] == len(feature_cols)


def test_train_network():
    rng = np.random.default_rng(7)
    samples = 200
    x = rng.normal(size=(2, samples))
    y = (x[0] + x[1] > 0).astype(int).reshape(1, samples)

    layer_dims = [2, 4, 1]
    weights, biases, _ = train_network(
        x,
        y,
        layer_dims,
        epochs=1000,
        learning_rate=0.1,
    )

    preds = predict(x, weights, biases)
    pred_labels = (preds >= 0.5).astype(int)
    accuracy = np.mean(pred_labels == y)
    assert accuracy > 0.9


if __name__ == "__main__":
    test_data_prep()
    test_train_network()
    print("All smoke tests passed.")
