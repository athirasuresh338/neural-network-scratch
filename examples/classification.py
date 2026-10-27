import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import NeuralNetwork
from train import train
from loss import binary_cross_entropy


def get_dataset_path() -> str:
    """Locates the dataset CSV in the project root directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for filename in ["student_performance.csv", "student_performance_sample.csv"]:
        path = os.path.join(project_root, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate dataset in {project_root}")


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into train and test sets using NumPy."""
    if seed is not None:
        np.random.seed(seed)
    m = len(X)
    shuffled_idx = np.random.permutation(m)
    test_size = int(m * test_ratio)
    test_idx = shuffled_idx[:test_size]
    train_idx = shuffled_idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardizes features to zero mean and unit variance.
    Statistics are computed ONLY from X_train to prevent data leakage.
    """
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0.0] = 1.0  # Prevent divide-by-zero

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    return X_train_scaled, X_test_scaled


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates classification metrics manually using NumPy:
    Accuracy, Precision, Recall, and F1-Score.
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    tp = int(np.sum((y_pred_flat == 1) & (y_true_flat == 1)))
    tn = int(np.sum((y_pred_flat == 0) & (y_true_flat == 0)))
    fp = int(np.sum((y_pred_flat == 1) & (y_true_flat == 0)))
    fn = int(np.sum((y_pred_flat == 0) & (y_true_flat == 1)))

    total = len(y_true_flat)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_pipeline() -> Dict[str, Any]:
    """Full workflow: load, split, scale, train, and evaluate."""
    csv_path = get_dataset_path()
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 5 input features (final_grade is excluded to prevent target leakage)
    features = [
        "study_hours",
        "attendance_percentage",
        "previous_grade",
        "sleep_hours",
        "absences",
    ]
    target = "pass"

    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # 1. Train / test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    print(f"Dataset: {len(X)} records | Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. Standardization (strictly using training statistics)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    # 3. Model: [5, 8, 4, 1]
    model = NeuralNetwork(layer_sizes=[5, 8, 4, 1], seed=42)

    # 4. Training (Full-batch gradient descent)
    epochs = 1000
    learning_rate = 0.05
    print(f"Training for {epochs} epochs (learning_rate={learning_rate})...")
    loss_history = train(model, X_train_scaled, y_train, epochs=epochs, learning_rate=learning_rate, print_every=200)

    # 5. Evaluate on unseen test data
    test_probs = model.predict_proba(X_test_scaled)
    test_preds = model.predict(X_test_scaled, threshold=0.5)
    test_loss = binary_cross_entropy(y_test, test_probs)
    metrics = evaluate_predictions(y_test, test_preds)

    print("\n" + "=" * 48)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 48)
    print(f"Test Loss (BCE):    {test_loss:.5f}")
    print(f"Accuracy:           {metrics['accuracy'] * 100:.2f}% ({metrics['tp'] + metrics['tn']}/{len(y_test)} correct)")
    print(f"Precision:          {metrics['precision'] * 100:.2f}%")
    print(f"Recall:             {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:           {metrics['f1'] * 100:.2f}%")
    print(f"Majority Baseline:  {36 / 50 * 100:.2f}% (Predicting all 0s)")
    print("\nConfusion Matrix:")
    print(f"  - True Negatives  (TN): {metrics['tn']} (Correct Fail)")
    print(f"  - False Positives (FP): {metrics['fp']} (Spurious Pass)")
    print(f"  - False Negatives (FN): {metrics['fn']} (Missed Pass)")
    print(f"  - True Positives  (TP): {metrics['tp']} (Correct Pass)")

    # Sample table
    print("\nSample Test Predictions:")
    print(f"{'Index':<6} {'Actual':<8} {'Predicted':<10} {'Probability':<12} {'Result'}")
    print("-" * 48)
    for i in range(10):
        actual = int(y_test[i, 0])
        pred = int(test_preds[i, 0])
        prob = float(test_probs[i, 0])
        status = "CORRECT" if actual == pred else "INCORRECT"
        print(f"{i:<6} {actual:<8} {pred:<10} {prob:<12.4f} {status}")

    return {
        "model": model,
        "loss_history": loss_history,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_loss": test_loss,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_pipeline()
