"""Loading, cleaning and splitting the UCI Bank Marketing dataset.

The one rule this module enforces: the test set is separated before anything is
learned from the data, and nothing downstream is allowed to look at it.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    LEAKY_COLUMNS,
    RANDOM_STATE,
    RAW_CSV,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV. It is semicolon-separated despite the .csv name."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python src/download_data.py` first."
        )
    return pd.read_csv(RAW_CSV, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the target into 0/1 and leave the features otherwise untouched.

    Note on "unknown": several categorical columns use the literal string
    "unknown" instead of a blank. We deliberately keep it as its own category
    rather than imputing it. Whether a bank knows a customer's job is itself
    information, and inventing a value would be a guess we cannot justify.
    """
    df = df.copy()
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def split(df: pd.DataFrame, drop_leaky: bool = True):
    """Split into train / validation / test, stratified on the target.

    train      - the model learns its parameters here
    validation - the decision threshold is chosen here
    test       - touched once, at the very end, for the reported numbers

    Stratifying keeps the same yes/no ratio in all three parts. Without it, a
    random split of an 11.7% positive class can hand one part noticeably more
    positives than another and make the results depend on luck.
    """
    features = df.drop(columns=[TARGET])
    target = df[TARGET]

    if drop_leaky:
        features = features.drop(columns=LEAKY_COLUMNS)

    X_rest, X_test, y_rest, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=VAL_SIZE,
        stratify=y_rest,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def column_types(X: pd.DataFrame):
    """Split column names into categorical (text) and numeric."""
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric
