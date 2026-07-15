"""Small real-library checks complementing the mocked end-to-end smoke test."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hr_recourse.counterfactuals import run_dice_level
from hr_recourse.data_modeling import (
    TARGET,
    build_preprocessor,
    choose_threshold,
    feature_columns,
    load_and_validate_data,
    split_data,
)
from hr_recourse.explainability import compute_shap_importance


def test_real_dice_and_shap_integrations():
    data, _ = load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    splits = split_data(data, seed=42)
    features = feature_columns(splits.train)
    model = Pipeline([
        ("preprocessor", build_preprocessor(splits.train, scale_numeric=True)),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42,
        )),
    ])
    model.fit(splits.train[features], splits.train[TARGET])

    validation_scores = model.predict_proba(splits.validation[features])[:, 1]
    threshold, _ = choose_threshold(splits.validation[TARGET], validation_scores)
    test_scored = splits.test.copy()
    test_scored["attrition_score"] = model.predict_proba(test_scored[features])[:, 1]
    cohort = test_scored.loc[test_scored["EmployeeNumber"].eq(144)].copy()
    assert len(cohort) == 1
    assert cohort.iloc[0]["attrition_score"] >= threshold

    for level in (1, 2):
        result = run_dice_level(
            splits.train, cohort, model, threshold, level,
            total_cfs=1, retry_count=1, seed=42,
        )
        assert result.status["success"].all(), result.status["error"].tolist()
        assert not result.scenarios.empty
        assert result.scenarios["model_valid"].any()
        assert result.scenarios["changes_json"].str.startswith("{").all()

    feature_importance, group_importance, local = compute_shap_importance(
        model, splits.train, splits.test.head(8), seed=42, max_background=20,
    )
    assert len(feature_importance) == 30
    assert set(group_importance["feature_group"]) == {
        "immutable", "historical", "perception_experience",
        "job_structure_context", "directly_actionable",
    }
    assert not feature_importance["feature_group"].eq("unassigned").any()
    assert len(local) == 8
