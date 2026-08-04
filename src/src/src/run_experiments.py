from __future__ import annotations

import itertools
import os
import shutil
import subprocess
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/train_config.yaml")
OUT_DIR = Path("configs/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

experiments = [
    {"model_type": "random_forest", "n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
    {"model_type": "random_forest", "n_estimators": 200, "max_depth": 8, "min_samples_split": 4, "min_samples_leaf": 2},
    {"model_type": "random_forest", "n_estimators": 300, "max_depth": 10, "min_samples_split": 4, "min_samples_leaf": 2},
    {"model_type": "random_forest", "n_estimators": 150, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    {"model_type": "random_forest", "n_estimators": 250, "max_depth": 12, "min_samples_split": 6, "min_samples_leaf": 3},
]

for i, overrides in enumerate(experiments, start=1):
    cfg = base.copy()
    cfg["model"] = {**cfg["model"], **overrides}
    cfg["project"]["experiment_name"] = base["project"]["experiment_name"]
    out = OUT_DIR / f"experiment_{i}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    subprocess.run(["python", "-m", "src.models.train", "--config", str(out)], check=True)
