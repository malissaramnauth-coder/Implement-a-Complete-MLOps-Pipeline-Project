import pytest
from src.evaluate import compute_metrics

def test_compute_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.6666666666666666


def test_perfect_prediction():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_all_predictions_negative():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    