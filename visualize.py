import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from examples.classification import run_pipeline


def plot_architecture(save_path: str):
    """
    Plots a clear schematic diagram of the neural network architecture [5, 8, 4, 1].
    Layout is carefully spaced so titles, layer labels, nodes, and feature names never overlap.
    """
    layer_sizes = [5, 8, 4, 1]
    layer_names = [
        "Input Layer\n(5 features)",
        "Hidden Layer 1\n(8 neurons, ReLU)",
        "Hidden Layer 2\n(4 neurons, ReLU)",
        "Output Layer\n(1 neuron, Sigmoid)",
    ]
    feature_labels = [
        "study_hours",
        "attendance",
        "previous_grade",
        "sleep_hours",
        "absences",
    ]

    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=300)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0.0, 1.14)
    ax.axis("off")

    x_coords = [0.12, 0.40, 0.68, 0.92]
    node_positions = []

    # Coordinates for each neuron in each layer
    for layer_idx, num_nodes in enumerate(layer_sizes):
        y_coords = np.linspace(0.10, 0.78, num_nodes)
        node_positions.append([(x_coords[layer_idx], y) for y in y_coords])

    # Connection lines between adjacent layers
    for l in range(len(layer_sizes) - 1):
        for (x1, y1) in node_positions[l]:
            for (x2, y2) in node_positions[l + 1]:
                ax.plot([x1, x2], [y1, y2], color="#94a3b8", alpha=0.35, linewidth=0.8, zorder=1)

    layer_colors = ["#38bdf8", "#818cf8", "#a78bfa", "#f43f5e"]

    # Draw neuron circles and labels
    for layer_idx, nodes in enumerate(node_positions):
        color = layer_colors[layer_idx]
        for node_idx, (x, y) in enumerate(nodes):
            ax.scatter(
                [x], [y],
                s=700,
                color=color,
                edgecolors="#1e293b",
                linewidths=1.5,
                zorder=3
            )

            # Feature names on left of input layer
            if layer_idx == 0:
                ax.text(
                    x - 0.05,
                    y,
                    feature_labels[node_idx],
                    ha="right",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#0f172a",
                )
            # Output label on right of output node
            elif layer_idx == len(layer_sizes) - 1:
                ax.text(
                    x + 0.05,
                    y,
                    r"$\hat{y} = P(\mathrm{pass})$",
                    ha="left",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color="#0f172a",
                )

        # Layer subtitle
        ax.text(
            x_coords[layer_idx],
            0.90,
            layer_names[layer_idx],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
        )

    # Main title
    ax.text(
        0.50,
        1.06,
        "Fully Connected Feedforward Neural Network Architecture [5, 8, 4, 1]",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_curve(loss_history: list, save_path: str):
    """
    Plots the training loss curve across epochs.
    Y-axis adapts dynamically to actual loss values.
    """
    epochs = range(1, len(loss_history) + 1)
    min_loss, max_loss = min(loss_history), max(loss_history)
    loss_range = max_loss - min_loss

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    ax.plot(epochs, loss_history, color="#2563eb", linewidth=2.2, label="Binary Cross-Entropy Loss")

    # Dynamic y-axis margins (adds 10% headroom above and below)
    ax.set_ylim(min_loss - 0.1 * loss_range, max_loss + 0.12 * loss_range)

    # Annotate starting loss
    ax.scatter([1], [loss_history[0]], color="#dc2626", s=55, zorder=5)
    ax.text(
        25,
        loss_history[0] + 0.02 * loss_range,
        f"Epoch 1 Loss: {loss_history[0]:.4f}",
        fontsize=9.5,
        fontweight="bold",
        color="#dc2626",
    )

    # Annotate final loss
    ax.scatter([len(epochs)], [loss_history[-1]], color="#16a34a", s=55, zorder=5)
    ax.text(
        len(epochs) - 220,
        loss_history[-1] + 0.04 * loss_range,
        f"Final Loss: {loss_history[-1]:.4f}",
        fontsize=9.5,
        fontweight="bold",
        color="#16a34a",
    )

    ax.set_title("Training Loss Convergence (Full-Batch Gradient Descent)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_actual_vs_predicted(y_test: np.ndarray, test_probs: np.ndarray, save_path: str):
    """
    Compares actual test labels with predicted probabilities.
    """
    y_true_flat = y_test.ravel()
    y_prob_flat = test_probs.ravel()
    num_samples = len(y_true_flat)
    sample_indices = np.arange(num_samples)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # Decision boundary at threshold = 0.5
    ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")

    pass_mask = y_true_flat == 1
    fail_mask = y_true_flat == 0

    ax.scatter(
        sample_indices[fail_mask],
        y_prob_flat[fail_mask],
        color="#dc2626",
        edgecolor="#7f1d1d",
        s=60,
        label=f"Actual Fail (0) [n={np.sum(fail_mask)}]",
        alpha=0.85,
    )
    ax.scatter(
        sample_indices[pass_mask],
        y_prob_flat[pass_mask],
        color="#16a34a",
        edgecolor="#14532d",
        s=70,
        marker="^",
        label=f"Actual Pass (1) [n={np.sum(pass_mask)}]",
        alpha=0.9,
    )

    ax.set_ylim(-0.08, 1.22)
    ax.set_title(
        "Test Set Predictions: Predicted Probability vs. Actual Label",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Test Record Index", fontsize=10.5)
    ax.set_ylabel(r"Predicted Probability $\hat{y} = P(\mathrm{pass})$", fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Architecture diagram
    arch_path = os.path.join(plots_dir, "neural_network_architecture.png")
    plot_architecture(arch_path)

    # 2. Run pipeline to gather metrics
    print("\nExecuting classification pipeline...")
    results = run_pipeline()

    # 3. Loss curve (dynamically scaled)
    loss_path = os.path.join(plots_dir, "training_loss_curve.png")
    plot_loss_curve(results["loss_history"], loss_path)

    # 4. Actual vs predicted
    pred_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plot_actual_vs_predicted(results["y_test"], results["test_probs"], pred_path)

    print("\nAll 3 visualization plots regenerated successfully!")


if __name__ == "__main__":
    main()
