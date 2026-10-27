import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes mean Binary Cross-Entropy loss across m samples.
    Predictions are clipped to [1e-15, 1 - 1e-15] to prevent log(0).
    """
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(loss)


def output_layer_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes error gradient dZ at the output layer: dL / dZ = y_pred - y_true.
    Derived from combining BCE derivative with the Sigmoid derivative.
    """
    y_true = np.asarray(y_true).reshape(-1, 1)
    return y_pred - y_true
