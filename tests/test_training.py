import json

import pandas as pd
import pytest

from online_shoppers_intention import models
from online_shoppers_intention.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PAGE_VALUES,
    TARGET,
    holdout_indices,
    split_features_target,
)
from online_shoppers_intention.models import make_preprocessor
from online_shoppers_intention.train import main


@pytest.fixture
def sessions():
    return pd.DataFrame(
        {
            **{
                column: [float(i % 4) for i in range(60)] for column in NUMERIC_FEATURES
            },
            **{column: [i % 2 for i in range(60)] for column in CATEGORICAL_FEATURES},
            "Month": ["Feb", "Mar"] * 30,
            "VisitorType": ["Returning_Visitor", "New_Visitor"] * 30,
            "Weekend": [False, True] * 30,
            PAGE_VALUES: [0.0, 10.0] * 30,
            TARGET: ["FALSE", "TRUE"] * 30,
        }
    )


def test_feature_selection_excludes_target_and_page_values(sessions):
    features, target = split_features_target(sessions.assign(unrelated=1))

    assert list(features.columns) == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert target.tolist() == [0, 1] * 30
    diagnostic, _ = split_features_target(sessions, include_page_values=True)
    assert list(diagnostic.columns) == [*features.columns, PAGE_VALUES]


def test_missing_page_values_affects_only_diagnostic(sessions):
    clean = sessions.drop(columns=PAGE_VALUES)

    split_features_target(clean)
    with pytest.raises(ValueError, match="Missing required CSV columns: PageValues"):
        split_features_target(clean, include_page_values=True)


def test_split_rejects_invalid_revenue(sessions):
    sessions.loc[0, TARGET] = "unknown"

    with pytest.raises(ValueError, match="Revenue"):
        split_features_target(sessions)


def test_holdout_is_disjoint_stratified_and_reproducible(sessions):
    _, target = split_features_target(sessions)
    train, test = holdout_indices(target)
    repeated_train, repeated_test = holdout_indices(target)

    assert train.intersection(test).empty
    assert len(train) == 48
    assert len(test) == 12
    assert target.loc[train].mean() == target.loc[test].mean() == 0.5
    assert train.equals(repeated_train)
    assert test.equals(repeated_test)


def test_preprocessing_handles_missing_numbers_and_unseen_categories(sessions):
    features, _ = split_features_target(sessions)
    preprocessor = make_preprocessor(scale_numeric=True).fit(features)
    row = features.iloc[[0]].assign(Month="Nov", Administrative=float("nan"))
    transformed = preprocessor.transform(row)

    assert transformed.shape == (1, len(preprocessor.get_feature_names_out()))
    assert not pd.isna(transformed).any()
    assert all(PAGE_VALUES not in name for name in preprocessor.get_feature_names_out())


def test_cli_trains_from_local_data_and_reports_diagnostic(
    sessions, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(models, "LOGISTIC_GRID", {"classifier__C": [1.0]})
    monkeypatch.setattr(models, "TREE_GRID", {"classifier__max_depth": [2]})
    csv = tmp_path / "sessions.csv"
    sessions.to_csv(csv, index=False)

    main(["--data", str(csv), "--diagnose-page-values"])

    report = json.loads(capsys.readouterr().out)
    assert report["train_sessions"] == 48
    assert report["holdout_sessions"] == 12
    assert report["selected_model"] in report["models"]
    for name, model in report["models"].items():
        assert model["holdout"]["confusion_matrix"] == [[6, 0], [0, 6]]
        diagnostic = report["page_values_training_cv_diagnostic"][name]
        assert diagnostic["cv_pr_auc_without_page_values"] == model["cv_pr_auc"]
