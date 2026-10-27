import numpy as np


class Dense:
    """
    A fully connected (dense) layer.
    
    Attributes:
        W (np.ndarray): Weight matrix of shape (n_in, n_out).
        b (np.ndarray): Bias row vector of shape (1, n_out).
        X (np.ndarray): Cached input matrix from the forward pass.
        dW (np.ndarray): Gradient of loss with respect to W.
        db (np.ndarray): Gradient of loss with respect to b.
    """

    def __init__(self, n_in: int, n_out: int):
        # He (Kaiming) normal initialization for stable variance with ReLU
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Z = X @ W + b
        """
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Backward pass:
            dW = (1 / m) * (X^T @ dZ)
            db = (1 / m) * sum(dZ, axis=0)
            dX = dZ @ W^T
        """
        m = self.X.shape[0]
        self.dW = (1.0 / m) * (self.X.T @ dZ)
        self.db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T
