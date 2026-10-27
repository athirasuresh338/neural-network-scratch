import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    """ReLU activation: f(z) = max(0, z)"""
    return np.maximum(0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 for z > 0, 0 otherwise."""
    return np.where(Z > 0, 1.0, 0.0)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: sigma(z) = 1 / (1 + exp(-z)).
    Clipped to [-500, 500] to prevent overflow.
    """
    Z_clipped = np.clip(Z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-Z_clipped))
