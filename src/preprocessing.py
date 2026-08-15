# write a program that splits my dataset into train and test sets and saves them to the specified paths in the config file. The program should read the dataset from the path specified in the config file, split it into train and test sets, and save them to the paths specified in the config file. The program should also log the number of rows in the train and test sets.
# Here's a Python program that reads a dataset from a specified path in a configuration file, splits it into training and testing sets, saves them to the specified paths, and logs the number of rows in each set. This program uses the `pandas` library for data manipulation and `yaml` for reading the configuration file.
import pandas as pd
import yaml
import os
import logging
from typing import Iterable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load the raw dataset from *data_path* as a DataFrame."""
    df = pd.read_csv(data_path)
    logging.info(f"Loaded raw dataset with {df.shape[0]} rows from {data_path}.")
    return df


def split_data(config):
    # Load dataset
    df = pd.read_csv(config['data']['dataset_path'])
    logging.info(f"Loaded dataset with {df.shape[0]} rows.")

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logging.info(f"Split dataset into {train_df.shape[0]} train rows and {test_df.shape[0]} test rows.")

    # Save split datasets
    train_df.to_csv(config['data']['train_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)
    logging.info("Saved split datasets.")

def validate_input(df: pd.DataFrame, target: str) -> None:
    """Validate a raw dataframe before preprocessing.

    Raises ``TypeError`` if *df* is not a DataFrame, and ``ValueError`` if it is
    empty or is missing the target column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not found. Available columns: "
            f"{list(df.columns)}"
        )

def encode_categoricals(
    df: pd.DataFrame, categorical_columns: Iterable[str]
) -> pd.DataFrame:
    """Return a copy of *df* with the given categorical columns one-hot encoded.

    Non-categorical columns are passed through unchanged. Input is not mutated.
    """
    categorical_columns = list(categorical_columns)
    missing = [c for c in categorical_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found for encoding: {missing}")
    return pd.get_dummies(df, columns=categorical_columns, dtype=int)

def prepare_features(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into feature matrix ``X`` and binary target ``y``.

    Drops configured constant/identifier columns and maps the positive label to
    1. Does not mutate the input.
    """
    data_cfg = config["data"]
    target = data_cfg["target"]
    validate_input(df, target)

    work = df.drop(columns=data_cfg.get("drop_columns", []), errors="ignore")
    y = (work[target] == data_cfg["positive_label"]).astype(int)
    X = work.drop(columns=[target])
    return X, y

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
) -> ColumnTransformer:
    """Assemble the ColumnTransformer used inside the training pipeline."""
    pcfg = config["preprocessing"]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=pcfg.get("numeric_imputer", "median")))
    ]
    if pcfg.get("scale_numeric", True):
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=pcfg.get("categorical_imputer", "most_frequent")
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=pcfg.get("onehot_handle_unknown", "ignore")
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

if __name__ == "__main__":
    config = load_config("configs/model.yaml")
    split_data(config)
