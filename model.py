import numpy as np
from typing import List
from layers import Dense
from activations import relu, relu_derivative, sigmoid
from loss import output_layer_gradient


class NeuralNetwork:
    """
    Configurable fully connected feedforward neural network for binary classification.
    Hidden layers use ReLU; the single output neuron uses Sigmoid.
    """

    def __init__(self, layer_sizes: List[int], seed: int = None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output layer sizes.")
        if layer_sizes[-1] != 1:
            raise ValueError(f"Output layer must have 1 neuron for binary classification, got {layer_sizes[-1]}.")

        self.layer_sizes = layer_sizes

        if seed is not None:
            np.random.seed(seed)

        # Build dense layers between consecutive layer sizes
        self.layers: List[Dense] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1]))

        # Cached values from forward pass needed for backpropagation
        self.zs: List[np.ndarray] = []
        self.y_pred: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers:
        - Hidden layers: A = ReLU(Z) where Z = X @ W + b
        - Output layer:  y_pred = Sigmoid(Z)
        """
        self.zs = []
        A = X

        # Hidden layers with ReLU
        for layer in self.layers[:-1]:
            Z = layer.forward(A)
            self.zs.append(Z)
            A = relu(Z)

        # Final output layer with Sigmoid
        Z_out = self.layers[-1].forward(A)
        self.zs.append(Z_out)
        self.y_pred = sigmoid(Z_out)

        return self.y_pred

    def backward(self, y_true: np.ndarray) -> None:
        """
        Backpropagation using analytical chain rule:
        1. Output layer error: dZ = y_pred - y_true
        2. Hidden layer errors: dZ[l] = dA[l] * relu_derivative(Z[l])
        """
        # Output layer gradient (Sigmoid + BCE simplification)
        dZ = output_layer_gradient(y_true, self.y_pred)
        dA = self.layers[-1].backward(dZ)

        # Hidden layers in reverse order
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            Z = self.zs[i]
            dZ = dA * relu_derivative(Z)
            dA = layer.backward(dZ)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Gradient descent parameter update:
            W = W - learning_rate * dW
            b = b - learning_rate * db
        """
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probabilities in range (0, 1)."""
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 or 1) using the probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
