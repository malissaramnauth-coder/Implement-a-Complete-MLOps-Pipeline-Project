from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.metric_preset import DataDriftPreset

from src.data.dataset import load_raw_data


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_production_slice(df: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    current = df.sample(frac=fraction, random_state=random_state).copy()
    rng = np.random.default_rng(random_state)
    if "MonthlyIncome" in current.columns:
        current["MonthlyIncome"] = current["MonthlyIncome"] * 1.15
    if "DistanceFromHome" in current.columns:
        current["DistanceFromHome"] = current["DistanceFromHome"] + 2
    if "OverTime" in current.columns:
        mask = rng.random(len(current)) > 0.35
        current.loc[mask, "OverTime"] = "Yes"
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
    report.run(reference_data=reference, current_data=current)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))

    # Best-effort console summary compatible with Evidently's current report objects
    result = report.as_dict()
    overall = result["metrics"][0]["result"]
    share = overall.get("share_of_drifted_columns", 0.0)
    drifted = []
    for item in overall.get("drift_by_columns", []):
        if item.get("drift_detected"):
            drifted.append(item.get("column_name"))

    print(f"Share of drifted columns: {share:.3f}")
    print(f"Drifted columns: {drifted}")

    if share > threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
