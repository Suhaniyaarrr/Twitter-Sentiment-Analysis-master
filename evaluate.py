"""
Model evaluation utilities: accuracy, classification report, confusion matrix.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return a results dict.

    Returns:
        {
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float,
          "confusion_matrix": np.ndarray,
          "classification_report": str,
          "y_pred": np.ndarray,
        }
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=list(config.REVERSE_LABEL_MAP.values()),
        ),
        "y_pred": y_pred,
    }


def print_evaluation(model, X_test, y_test):
    """Print a human-readable evaluation summary."""
    res = evaluate_model(model, X_test, y_test)
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy  : {res['accuracy']:.4f}")
    print(f"Precision : {res['precision']:.4f}")
    print(f"Recall    : {res['recall']:.4f}")
    print(f"F1 Score  : {res['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(res["confusion_matrix"])
    print("\nClassification Report:")
    print(res["classification_report"])
    return res
