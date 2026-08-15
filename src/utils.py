import hashlib

import mlflow
import mlflow.sklearn


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point mlflow at the given tracking server and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def file_md5(path: str) -> str:
    """Return the MD5 hex digest of the file at *path*, used to version data."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def log_run_metadata(config: dict, data_version: str, metrics: dict, pipeline) -> None:
    """Log model config, data version, metrics, and the fitted pipeline to the active mlflow run."""
    mlflow.log_params(config.get("model", {}))
    mlflow.log_param("data_version", data_version)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, "model")
