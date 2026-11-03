import numpy as np

class Dense:
    """A fully connected neural network layer."""

    def __init__(self, input_dim: int, output_dim: int):
        # He initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

        self.X = None
        self.dW = None
        self.db = None


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: Z = XW + b."""
        self.X = X
        return X @ self.W + self.b


    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Compute gradients and return the gradient with respect to the input."""
        m = self.X.shape[0]

        self.dW = (self.X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        return dZ @ self.W.T