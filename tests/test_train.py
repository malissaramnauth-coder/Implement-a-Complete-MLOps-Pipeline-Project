import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.train import build_model

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