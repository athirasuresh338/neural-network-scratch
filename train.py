from typing import List
import numpy as np
from model import NeuralNetwork
from loss import binary_cross_entropy

def train(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    print_every: int = 100,
    verbose: bool = True) -> List[float]:
    """Train the neural network using full-batch gradient descent."""

    y = np.asarray(y).reshape(-1, 1)
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = model.forward(X)
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # Backward pass
        model.backward(y)

        # Update parameters
        model.update_parameters(learning_rate)

        # Print training progress
        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            print(f"Epoch {epoch:4d}/{epochs} - Loss: {loss:.4f}")
    return loss_history