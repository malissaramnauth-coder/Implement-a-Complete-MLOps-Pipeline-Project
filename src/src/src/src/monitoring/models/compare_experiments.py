from __future__ import annotations

import argparse

import mlflow


def main(experiment_name: str, metric: str) -> None:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs found")

    cols = [
        "run_id",
        f"metrics.{metric}",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.roc_auc",
        "params.model_type",
        "params.n_estimators",
        "params.max_depth",
        "params.data_version",
    ]
    cols = [c for c in cols if c in runs.columns]

    print(runs[cols].head(10).to_string(index=False))
    best = runs.iloc[0]
    print("\nBest run:")
    print(best[cols].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--metric", default="f1")
    args = parser.parse_args()
    main(args.experiment_name, args.metric)
