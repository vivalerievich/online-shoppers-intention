import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from online_shoppers_intention.data import (
    DEFAULT_DATA,
    holdout_indices,
    split_features_target,
)
from online_shoppers_intention.models import (
    CV_FOLDS,
    PR_AUC_SCORER,
    model_searches,
    training_cv,
    trapezoidal_pr_auc,
    with_page_values,
)


def evaluate(y_true, probabilities, *, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": trapezoidal_pr_auc(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def train(
    sessions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    diagnose_page_values: bool = False,
    model_out: Path | None = None,
) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    features, target = split_features_target(sessions)
    diagnostic_features = None
    if diagnose_page_values:
        diagnostic_features, _ = split_features_target(
            sessions, include_page_values=True
        )

    train_index, test_index = holdout_indices(target)
    x_train, y_train = features.loc[train_index], target.loc[train_index]
    x_test, y_test = features.loc[test_index], target.loc[test_index]
    if y_train.value_counts().min() < CV_FOLDS or y_test.nunique() != 2:
        raise ValueError(
            f"Need at least {CV_FOLDS} rows per class in train and both classes in holdout."
        )

    cv = list(training_cv().split(x_train, y_train))
    reports = {}
    diagnostics = {}
    fitted_models = {}
    for name, search in model_searches(cv).items():
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        probabilities = fitted_models[name].predict_proba(x_test)[:, 1]
        reports[name] = {
            "parameters": search.best_params_,
            "cv_pr_auc": search.best_score_,
            "holdout": evaluate(y_test, probabilities, threshold=threshold),
        }
        if diagnostic_features is not None:
            diagnostic_scores = cross_val_score(
                with_page_values(search.best_estimator_),
                diagnostic_features.loc[train_index],
                y_train,
                cv=cv,
                scoring=PR_AUC_SCORER,
                error_score="raise",
            )
            diagnostics[name] = {
                "cv_pr_auc_without_page_values": search.best_score_,
                "cv_pr_auc_with_page_values": float(diagnostic_scores.mean()),
            }

    selected_model = max(reports, key=lambda name: reports[name]["cv_pr_auc"])
    if model_out is not None:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=model_out.parent) as directory:
            staged = Path(directory) / model_out.name
            joblib.dump(fitted_models[selected_model], staged, compress=3)
            staged.replace(model_out)

    report = {
        "train_sessions": len(x_train),
        "holdout_sessions": len(x_test),
        "holdout_purchase_rate": float(y_test.mean()),
        "threshold": threshold,
        "selected_model": selected_model,
        "models": reports,
    }
    if diagnostics:
        report["page_values_training_cv_diagnostic"] = diagnostics
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate purchase-intention classifiers."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Path to the UCI CSV."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for label metrics.",
    )
    parser.add_argument(
        "--diagnose-page-values",
        action="store_true",
        help="Compare the selected hyperparameters with and without PageValues using training CV.",
    )
    parser.add_argument(
        "--model-out", type=Path, help="Save the selected fitted pipeline to this path."
    )
    args = parser.parse_args(argv)
    if args.model_out is not None and args.model_out.resolve() == args.data.resolve():
        parser.error("--model-out must differ from the training CSV path")
    try:
        report = train(
            pd.read_csv(args.data),
            threshold=args.threshold,
            diagnose_page_values=args.diagnose_page_values,
            model_out=args.model_out,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
