import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate binary cross-entropy loss."""
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1.0 - y_true) * np.log(1.0 - y_pred)
    )
    return float(loss)


def output_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Combined gradient of sigmoid and binary cross-entropy: dZ = y_pred - y_true."""
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return y_pred - y_true