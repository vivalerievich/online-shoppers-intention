from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from online_shoppers_intention.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PAGE_VALUES,
    RANDOM_STATE,
)

CV_FOLDS = 5
LOGISTIC_GRID = {
    "classifier__C": [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    ]
}
TREE_GRID = {
    "classifier__max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
    "classifier__min_samples_leaf": [1, 5, 10, 20, 50, 100],
}


def make_preprocessor(
    *, scale_numeric: bool, include_page_values: bool = False
) -> ColumnTransformer:
    numeric_columns = list(NUMERIC_FEATURES)
    if include_page_values:
        numeric_columns.append(PAGE_VALUES)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )


def trapezoidal_pr_auc(y_true, probabilities) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return auc(recall, precision)


PR_AUC_SCORER = make_scorer(trapezoidal_pr_auc, response_method="predict_proba")


def training_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def model_searches(cv, *, return_train_score: bool = False) -> dict[str, GridSearchCV]:
    logistic = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    tree = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    return {
        name: GridSearchCV(
            estimator,
            grid,
            scoring=PR_AUC_SCORER,
            cv=cv,
            return_train_score=return_train_score,
            error_score="raise",
        )
        for name, estimator, grid in (
            ("LogisticRegression", logistic, LOGISTIC_GRID),
            ("DecisionTree", tree, TREE_GRID),
        )
    }


def with_page_values(model: Pipeline) -> Pipeline:
    preprocessor = model.named_steps["preprocessor"]
    scale_numeric = "scaler" in preprocessor.named_transformers_["numeric"].named_steps
    return clone(model).set_params(
        preprocessor=make_preprocessor(
            scale_numeric=scale_numeric, include_page_values=True
        )
    )
