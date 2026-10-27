import numpy as np
from typing import List
from model import NeuralNetwork
from loss import binary_cross_entropy


def train(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    print_every: int = 100,
    verbose: bool = True
) -> List[float]:
    """
    Trains the neural network using full-batch gradient descent.
    
    Returns:
        List[float]: History of Binary Cross-Entropy losses for each epoch.
    """
    y = np.asarray(y).reshape(-1, 1)
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        # 1. Forward pass
        y_pred = model.forward(X)

        # 2. Compute loss
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # 3. Backward pass (calculates dW and db)
        model.backward(y)

        # 4. Update parameters
        model.update_parameters(learning_rate)

        # 5. Log progress
        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            print(f"Epoch {epoch:5d}/{epochs} - Loss: {loss:.6f}")

    return loss_history
