import json

import numpy as np

from hr_recourse.counterfactuals import (
    evaluate_level3, generate_level3_candidates, is_no_counterfactual_result,
)
from hr_recourse.data_modeling import load_and_validate_data


class ActionModel:
    """A deterministic score that improves with the three permitted levers."""
    def predict_proba(self, X):
        income = X["MonthlyIncome"].astype(float).to_numpy()
        stock = X["StockOptionLevel"].astype(float).to_numpy()
        overtime = X["OverTime"].eq("Yes").astype(float).to_numpy()
        score = np.clip(0.85 - income / 100000 - stock * 0.12 + overtime * 0.10, 0.01, 0.99)
        return np.column_stack([1 - score, score])


def test_level3_candidates_obey_direction_and_exclude_original():
    train, _ = load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    employee = train.iloc[0]
    candidates, cap = generate_level3_candidates(employee, train, [0, .05, .10, .30])
    assert not candidates.empty
    assert (candidates["MonthlyIncome"] >= employee["MonthlyIncome"]).all()
    assert (candidates["MonthlyIncome"] <= cap + 1e-7).all()
    assert (candidates["MonthlyIncome"] <= employee["MonthlyIncome"] * 1.30 + .01).all()
    assert (candidates["StockOptionLevel"] >= employee["StockOptionLevel"]).all()
    assert not ((candidates["MonthlyIncome"] == employee["MonthlyIncome"])
                & (candidates["StockOptionLevel"] == employee["StockOptionLevel"])
                & (candidates["OverTime"] == employee["OverTime"])).any()
    if employee["OverTime"] == "No":
        assert candidates["OverTime"].eq("No").all()


def test_evaluated_level3_never_marks_constraint_violation_feasible():
    train, _ = load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    scenarios = evaluate_level3(train.iloc[0], train, ActionModel(), .70, [0, .05, .10, .30])
    feasible = scenarios[scenarios["hr_feasible"]]
    assert feasible["model_valid"].all()
    assert feasible["violations"].eq("").all()
    assert scenarios["changed_features"].str.split("|").apply(set).apply(
        lambda fields: fields.issubset({"MonthlyIncome", "StockOptionLevel", "OverTime"})
    ).all()
    for _, row in scenarios.iterrows():
        changes = json.loads(row["changes_json"])
        assert set(changes) == set(row["changed_features"].split("|"))
        assert all(set(values) == {"before", "after"} for values in changes.values())


def test_dice_no_result_is_not_a_technical_failure_message():
    assert is_no_counterfactual_result(
        "No counterfactuals found for any of the query points! Kindly check your configuration."
    )
    assert not is_no_counterfactual_result("model prediction failed")
