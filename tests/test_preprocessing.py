import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessor,
    encode_categoricals,
    prepare_features,
    split_data,
)


def test_encode_categoricals_one_hot_encodes_column():
    df = pd.DataFrame({"color": ["red", "blue", "red"], "size": [1, 2, 3]})
    encoded = encode_categoricals(df, ["color"])
    assert "color_red" in encoded.columns
    assert "color_blue" in encoded.columns
    assert encoded["color_red"].tolist() == [1, 0, 1]


def test_encode_categoricals_preserves_non_categorical_columns():
    df = pd.DataFrame({"color": ["red", "blue"], "size": [1, 2]})
    encoded = encode_categoricals(df, ["color"])
    assert encoded["size"].tolist() == [1, 2]
    assert "color" not in encoded.columns


def test_encode_categoricals_missing_column_raises():
    df = pd.DataFrame({"size": [1, 2]})
    with pytest.raises(ValueError):
        encode_categoricals(df, ["color"])


def test_prepare_features_splits_and_drops_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [30, 40, 50],
            "target": ["yes", "no", "yes"],
        }
    )
    config = {
        "data": {
            "target": "target",
            "positive_label": "yes",
            "drop_columns": ["id"],
        }
    }
    X, y = prepare_features(df, config)
    assert "id" not in X.columns
    assert "target" not in X.columns
    assert y.tolist() == [1, 0, 1]


def test_build_preprocessor_transforms_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "age": [30.0, None, 50.0],
            "color": ["red", "blue", "red"],
        }
    )
    config = {"preprocessing": {}}
    preprocessor = build_preprocessor(["age"], ["color"], config)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 1


def test_split_data_writes_train_and_test_csv(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    df = pd.DataFrame({"a": range(20), "b": range(20)})
    df.to_csv(dataset_path, index=False)

    config = {
        "data": {
            "dataset_path": str(dataset_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
    }
    split_data(config)

    assert train_path.exists()
    assert test_path.exists()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) + len(test_df) == 20
