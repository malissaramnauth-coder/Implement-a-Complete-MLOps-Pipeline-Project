"""Evaluation metrics for the heart disease classifier."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred):
    """Return a dict of classification metrics.

    Recall is the primary metric: in a medical screen, missing a sick
    patient (a false negative) is costlier than a false alarm.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }