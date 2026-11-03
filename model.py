import numpy as np
from typing import List
from layers import Dense
from activations import relu, relu_derivative, sigmoid
from loss import output_gradient

class NeuralNetwork:
    """Feedforward neural network for binary classification."""

    def __init__(self, layer_sizes: List[int]):
        if len(layer_sizes) < 2:
            raise ValueError("At least an input and output layer are required.")

        if layer_sizes[-1] != 1:
            raise ValueError("Output layer must have 1 neuron for binary classification.")

        self.layer_sizes = layer_sizes

        self.layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

        self.zs = []
        self.y_pred = None


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers, caching pre-activations."""

        self.zs = []
        A = X

        for layer in self.layers[:-1]:
            Z = layer.forward(A)
            self.zs.append(Z)
            A = relu(Z)

        Z = self.layers[-1].forward(A)
        self.zs.append(Z)

        self.y_pred = sigmoid(Z)
        return self.y_pred


    def backward(self, y_true: np.ndarray) -> None:
        """Backpropagate gradients using the chain rule."""

        dZ = output_gradient(y_true, self.y_pred)
        dA = self.layers[-1].backward(dZ)

        for i in reversed(range(len(self.layers) - 1)):
            dZ = dA * relu_derivative(self.zs[i])
            dA = self.layers[i].backward(dZ)


    def update_parameters(self, learning_rate: float) -> None:
        """Update weights and biases using gradient descent."""

        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities."""
        return self.forward(X)


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)