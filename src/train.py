from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.process_data import split_data 
from src.evaluate import compute_metrics 
from src.preprocess import build_preprocessor
from src.utils.mlflow_utils import configure_mlflow, file_md5, log_run_metadata


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model_cfg = config["model"]
    model_type = model_cfg.get("model_type", "random_forest")
    rs = config["data"].get("random_state", 42)

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            class_weight=model_cfg.get("class_weight", None),
            random_state=rs,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=model_cfg.get("class_weight", None), random_state=rs)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            max_depth=model_cfg.get("max_depth", 3),
            random_state=rs,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def main(config_path: str) -> None:
    config = load_config(config_path)
    configure_mlflow(config["mlflow"]["tracking_uri"], config["project"]["experiment_name"])

    X_train, X_test, y_train, y_test = load_and_prepare_data(config_path)
    preprocessor = build_preprocessor(X_train)
    model = build_model(config)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_classification_model(pipeline, X_test, y_test)
        data_version = file_md5(config["data"]["raw_data_path"])
        log_run_metadata(config, data_version, metrics, pipeline)
        validate_metric_thresholds(metrics, config["metrics"])

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
