import pandas as pd
import pytest

from src.preprocessing import validate_input


def test_validate_input_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_input([1, 2, 3], target="target")


def test_validate_input_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["target"])
    with pytest.raises(ValueError):
        validate_input(df, target="target")


def test_validate_input_rejects_missing_target_column():
    df = pd.DataFrame({"age": [30, 40]})
    with pytest.raises(ValueError):
        validate_input(df, target="target")
