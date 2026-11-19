import argparse
import sys
from pathlib import Path
from pickle import UnpicklingError

import joblib
import pandas as pd

from online_shoppers_intention.data import select_features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score purchase intention from a feature CSV."
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Trusted joblib pipeline."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="CSV containing session features."
    )
    args = parser.parse_args(argv)
    try:
        features = select_features(pd.read_csv(args.data))
        model = joblib.load(args.model)
        probabilities = model.predict_proba(features)[:, 1]
    except (EOFError, OSError, UnpicklingError, ValueError) as error:
        parser.error(str(error))
    pd.DataFrame({"purchase_probability": probabilities}).to_csv(
        sys.stdout, index=False
    )


if __name__ == "__main__":
    main()
