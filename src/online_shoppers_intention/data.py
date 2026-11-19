from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA = Path("data/online_shoppers_intention.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Revenue"
PAGE_VALUES = "PageValues"

NUMERIC_FEATURES = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
)
CATEGORICAL_FEATURES = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)


def split_features_target(
    sessions: pd.DataFrame, *, include_page_values: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    if include_page_values:
        columns.append(PAGE_VALUES)
    missing = set([*columns, TARGET]).difference(sessions.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    target = (
        sessions[TARGET]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"false": 0, "true": 1, "0": 0, "1": 1})
    )
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Revenue must contain both purchase and non-purchase labels.")
    return sessions.loc[:, columns], target.astype(int)


def holdout_indices(target: pd.Series) -> tuple[pd.Index, pd.Index]:
    return train_test_split(
        target.index, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )
