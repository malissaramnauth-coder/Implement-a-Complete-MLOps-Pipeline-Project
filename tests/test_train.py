import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.train import (
    build_model,
    evaluate_classification_model,
    load_and_prepare_data,
    validate_metric_thresholds,
)

def test_build_random_forest():
   config = {
        "model": {
            "model_type": "random_forest"
        },
        "data": {
            "random_state": 42
        }
    }
   model=build_model(config)
   assert isinstance(model, RandomForestClassifier)

   
def test_build_logistic_regression():
    config = {
        "model": {
            "model_type": "logistic_regression"
        },
        "data": {
            "random_state": 42
        }
    }
    model = build_model(config)
    assert isinstance(model, LogisticRegression)


def test_build_gradient_boosting():
        config = {
            "model": {
                "model_type": "gradient_boosting"
            },
            "data": {
                "random_state": 42
            }
        }
        model = build_model(config)
        assert isinstance(model, GradientBoostingClassifier)


def _write_dataset(path, n=20):
    df = pd.DataFrame(
        {
            "id": range(n),
            "age": [30 + i for i in range(n)],
            "color": ["red", "blue"] * (n // 2),
            "target": ["yes", "no"] * (n // 2),
        }
    )
    df.to_csv(path, index=False)


def test_load_and_prepare_data_returns_train_test_splits(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    _write_dataset(dataset_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  dataset_path: {dataset_path}
  train_path: {tmp_path / 'train.csv'}
  test_path: {tmp_path / 'test.csv'}
  target: target
  positive_label: yes
  drop_columns: [id]
"""
    )

    X_train, X_test, y_train, y_test = load_and_prepare_data(str(config_path))

    assert "id" not in X_train.columns
    assert "target" not in X_train.columns
    assert set(y_train.unique()) <= {0, 1}
    assert len(X_train) + len(X_test) == 20


def test_evaluate_classification_model_returns_metrics():
    class StubPipeline:
        def predict(self, X):
            return [0, 1, 1, 0]

    metrics = evaluate_classification_model(StubPipeline(), X_test=None, y_test=[0, 1, 0, 0])
    assert metrics["accuracy"] == 0.75


def test_validate_metric_thresholds_raises_when_below_minimum():
    with pytest.raises(ValueError):
        validate_metric_thresholds({"recall": 0.5}, {"recall": 0.6})


def test_validate_metric_thresholds_passes_when_met():
    validate_metric_thresholds({"recall": 0.7, "accuracy": 0.8}, {"recall": 0.6})