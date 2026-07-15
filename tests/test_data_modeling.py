import numpy as np

from hr_recourse.data_modeling import (
    EXCLUDED_FEATURES, ID_COLUMN, TARGET, build_preprocessor, choose_threshold,
    feature_columns, load_and_validate_data, split_data,
)


def test_stratified_splits_are_disjoint_and_exclusions_hold():
    df, _ = load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    splits = split_data(df, seed=42)
    ids = [set(frame[ID_COLUMN]) for frame in (splits.train, splits.validation, splits.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert not set(EXCLUDED_FEATURES).intersection(feature_columns(df))
    assert TARGET not in feature_columns(df)


def test_threshold_tie_break_prefers_recall_then_precision():
    # At 0.50: TP=1, FP=0, FN=1. At 0.40: TP=2, FP=8, FN=0.
    # Both thresholds have the same F2, so the higher-recall 0.40 must win.
    y = np.array([1, 1, *([0] * 8)])
    scores = np.array([0.9, *([0.4] * 9)])
    threshold, table = choose_threshold(y, scores, minimum=0.4, maximum=0.5, step=0.1)
    rows = table.set_index("threshold")
    assert np.isclose(rows.loc[0.4, "f2"], rows.loc[0.5, "f2"])
    assert rows.loc[0.4, "recall"] > rows.loc[0.5, "recall"]
    assert threshold == 0.4


def test_preprocessor_learns_categories_from_train_only_and_handles_unknowns():
    df, _ = load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    train = df.iloc[:20].copy()
    validation = df.iloc[[20]].copy()
    train["JobRole"] = "train-only-role"
    validation["JobRole"] = "validation-only-role"
    features = feature_columns(train)
    preprocessor = build_preprocessor(train, scale_numeric=True)
    preprocessor.fit(train[features])

    categorical = [column for column in features if train[column].dtype == "object"]
    onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    job_role_categories = set(onehot.categories_[categorical.index("JobRole")])
    assert job_role_categories == {"train-only-role"}
    assert preprocessor.transform(validation[features]).shape[0] == 1
