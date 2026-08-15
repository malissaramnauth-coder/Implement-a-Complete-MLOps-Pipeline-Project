from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from src.preprocessing import load_raw_data


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_production_slice(df: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    current = df.sample(frac=fraction, random_state=random_state).copy()
    rng = np.random.default_rng(random_state)
    if "chol" in current.columns:
        current["chol"] = current["chol"] * 1.15
    if "trestbps" in current.columns:
        current["trestbps"] = current["trestbps"] + 10
    if "exang" in current.columns:
        mask = rng.random(len(current)) > 0.35
        current.loc[mask, "exang"] = 1
    return current


def main(config_path: str) -> None:
    config = load_config(config_path)
    data_path = config["data"]["raw_data_path"]
    report_path = Path(config["drift"]["report_path"])
    threshold = config["drift"]["prediction_drift_threshold"]
    fraction = config["drift"].get("production_fraction", 0.25)
    random_state = config["drift"].get("random_state", 42)

    reference = load_raw_data(data_path)
    current = make_production_slice(reference, fraction=fraction, random_state=random_state)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference),
        current_data=Dataset.from_pandas(current),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    # Best-effort console summary compatible with Evidently's current report objects
    metrics = snapshot.dict()["metrics"]
    share = 0.0
    drifted = []
    for item in metrics:
        name = item.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            share = item["value"].get("share", 0.0)
        elif name.startswith("ValueDrift(column="):
            column = item["config"]["column"]
            metric_threshold = item["config"].get("threshold", 0.05)
            is_p_value = "p_value" in item["config"].get("method", "")
            value = item["value"]
            column_drifted = value < metric_threshold if is_p_value else value > metric_threshold
            if column_drifted:
                drifted.append(column)

    print(f"Share of drifted columns: {share:.3f}")
    print(f"Drifted columns: {drifted}")

    if share > threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
