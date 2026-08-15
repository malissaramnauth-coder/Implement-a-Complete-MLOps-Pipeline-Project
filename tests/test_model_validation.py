from src.evaluate import compute_metrics


def test_compute_metrics_all_predictions_positive():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_compute_metrics_returns_expected_keys_and_types():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
