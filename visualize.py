import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from examples.classification import run_pipeline

# Configure Matplotlib for saving plots without a display
matplotlib.use("Agg")

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def plot_architecture(save_path: str):
    """Visualize the neural network architecture."""

    layer_sizes = [5, 8, 4, 1]

    layer_names = [
        "Input\n(5 features)",
        "Hidden 1\n(8 neurons, ReLU)",
        "Hidden 2\n(4 neurons, ReLU)",
        "Output\n(1 neuron, Sigmoid)"
    ]

    feature_names = [
        "study_hours",
        "attendance",
        "previous_grade",
        "sleep_hours",
        "absences"
    ]

    colors = ["#38bdf8", "#818cf8", "#a78bfa", "#f43f5e"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    x_positions = [0, 2.5, 5, 7.5]
    node_positions = []

    # Calculate neuron positions
    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 6, size)
        node_positions.append([(x_positions[i], y) for y in y_positions])

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        for x1, y1 in node_positions[i]:
            for x2, y2 in node_positions[i + 1]:
                ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.3, linewidth=0.8)

    # Draw neurons and labels
    for i, nodes in enumerate(node_positions):
        for j, (x, y) in enumerate(nodes):
            ax.scatter(x, y, s=600, color=colors[i], edgecolors="black", zorder=2)

            if i == 0:
                ax.text(x - 0.3, y, feature_names[j],
                        ha="right", va="center", fontsize=9)

            elif i == len(layer_sizes) - 1:
                ax.text(x + 0.3, y, r"$\hat{y} = P(\mathrm{pass})$",
                        ha="left", va="center", fontsize=10)

        ax.text(
            x_positions[i], 7,
            layer_names[i],
            ha="center", va="center",
            fontsize=10, fontweight="bold"
        )

    ax.set_title(
        "Feedforward Neural Network Architecture [5, 8, 4, 1]",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlim(-2.5, 10)
    ax.set_ylim(-1, 8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def plot_loss_curve(loss_history: list, save_path: str):
    """Plot training loss over epochs."""

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="#2563eb",
        linewidth=2,
        label="Binary Cross-Entropy Loss"
    )

    plt.title("Training Loss Curve", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    save_path: str
):
    """Compare actual test labels with predicted probabilities."""

    y_true = y_test.ravel()
    y_probs = test_probs.ravel()

    indices = np.arange(len(y_true))

    plt.figure(figsize=(10, 5))

    plt.scatter(
        indices[y_true == 0],
        y_probs[y_true == 0],
        color="#dc2626",
        marker="o",
        s=60,
        label="Actual Fail (0)"
    )

    plt.scatter(
        indices[y_true == 1],
        y_probs[y_true == 1],
        color="#16a34a",
        marker="^",
        s=60,
        label="Actual Pass (1)"
    )

    plt.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Decision Threshold (0.5)"
    )

    plt.title("Actual vs. Predicted Probabilities", fontsize=13)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Predicted Probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    """Run the classification pipeline and generate visualizations."""

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate architecture diagram
    plot_architecture(
        os.path.join(plots_dir, "neural_network_architecture.png")
    )

    # Run classification pipeline
    print("\nRunning classification pipeline...")
    results = run_pipeline()

    # Generate training and prediction plots
    plot_loss_curve(
        results["loss_history"],
        os.path.join(plots_dir, "training_loss_curve.png")
    )

    plot_actual_vs_predicted(
        results["y_test"],
        results["test_probs"],
        os.path.join(plots_dir, "actual_vs_predicted.png")
    )

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()

