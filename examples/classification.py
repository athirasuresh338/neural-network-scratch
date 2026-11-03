import os
import sys

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from model import NeuralNetwork
from train import train
from loss import binary_cross_entropy

def get_dataset_path():
    """Find the dataset in the project root directory."""

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for filename in ["student_performance.csv", "student_performance_sample.csv"]:
        path = os.path.join(project_root, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Dataset not found.")


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split the dataset into training and testing sets."""

    np.random.seed(seed)
    indices = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_features(X_train, X_test):
    """Standardize features using training set statistics only."""

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test


def evaluate_predictions(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1-score."""

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def run_pipeline():
    """Load data, train the network, and evaluate its performance."""

    # Load dataset
    df = pd.read_csv(get_dataset_path())

    features = [
        "study_hours",
        "attendance_percentage",
        "previous_grade",
        "sleep_hours",
        "absences"
    ]

    X = df[features].values
    y = df[["pass"]].values

    # Split and standardize data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = standardize_features(X_train, X_test)

    # Initialize and train model
    np.random.seed(42)
    model = NeuralNetwork([5, 8, 4, 1])

    loss_history = train(
        model,
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.05,
        print_every=200
    )

    # Evaluate model
    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)

    test_loss = binary_cross_entropy(y_test, test_probs)
    metrics = evaluate_predictions(y_test, test_preds)

    majority_class = np.bincount(y_train.ravel()).argmax()
    baseline = np.mean(y_test.ravel() == majority_class)

    print("\nTest Results")
    print("-" * 35)
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:  {metrics['precision'] * 100:.2f}%")
    print(f"Recall:     {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:   {metrics['f1'] * 100:.2f}%")
    print(f"Baseline:   {baseline * 100:.2f}%")

    print("\nConfusion Matrix")
    print(f"TN: {metrics['tn']}  FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}  TP: {metrics['tp']}")

    # Display sample predictions
    print("\nSample Predictions")

    for i in range(min(10, len(y_test))):
        print(
            f"Actual: {y_test[i, 0]} | "
            f"Predicted: {test_preds[i, 0]} | "
            f"Probability: {test_probs[i, 0]:.4f}"
        )

    return {
        "model": model,
        "loss_history": loss_history,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_loss": test_loss,
        "metrics": metrics
    }


if __name__ == "__main__":
    run_pipeline()