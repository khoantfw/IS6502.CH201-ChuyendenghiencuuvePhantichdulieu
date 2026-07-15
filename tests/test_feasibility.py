import pandas as pd

from hr_recourse.counterfactuals import SCENARIO_COLUMNS
from hr_recourse.feasibility import (
    audit_diagnostic_scenarios, build_assessment, rank_level3, violation_summary,
)


def scenario(employee, level, sid, valid, feasible, changes, n, salary=0, stock=0, score=.3):
    return {
        "employee_id": employee, "constraint_level": level, "scenario_id": sid,
        "score_before": .8, "score_after": score, "changed_features": changes,
        "n_changes": n, "salary_increase_pct": salary, "stock_option_increment": stock,
        "overtime_change": "none", "model_valid": valid, "hr_feasible": feasible,
        "violations": "",
    }


def test_locked_changes_are_violations_and_diagnostic_levels_not_ranked():
    raw = pd.DataFrame([scenario(1, 1, "broad", True, False, "Age|MonthlyIncome", 2)], columns=SCENARIO_COLUMNS)
    audited = audit_diagnostic_scenarios(raw)
    assert "immutable_changed" in audited.iloc[0]["violations"]
    mixed = pd.concat([audited, pd.DataFrame([
        scenario(1, 3, "l3", True, True, "OverTime", 1),
    ], columns=SCENARIO_COLUMNS)], ignore_index=True)
    assert rank_level3(mixed).iloc[0]["scenario_id"] == "l3"
    assert "diagnostic_level_not_selectable" not in set(violation_summary(audited)["violation"])


def test_ranking_and_exclusive_final_groups():
    cohort = pd.DataFrame([
        {"EmployeeNumber": 1, "attrition_score": .8},
        {"EmployeeNumber": 2, "attrition_score": .75},
        {"EmployeeNumber": 3, "attrition_score": .72},
    ])
    scenarios = pd.DataFrame([
        scenario(1, 3, "raise", True, True, "MonthlyIncome", 1, salary=5, score=.35),
        scenario(1, 3, "ot", True, True, "OverTime", 1, salary=0, score=.39),
        scenario(2, 1, "diagnostic", True, False, "Age", 1),
        scenario(3, 3, "invalid", False, False, "OverTime", 1, score=.6),
    ], columns=SCENARIO_COLUMNS)
    statuses = pd.DataFrame([
        {"employee_id": employee, "constraint_level": level, "success": True, "error": ""}
        for employee in (1, 2, 3) for level in (1, 2, 3)
    ])
    assessment, best = build_assessment(cohort, .5, scenarios, statuses)
    assert best.loc[best["employee_id"] == 1, "scenario_id"].iloc[0] == "ot"
    assert assessment.set_index("employee_id")["final_group"].to_dict() == {
        1: "actionable", 2: "model_valid_not_hr_feasible", 3: "no_model_recourse_within_search"
    }
    assert "diagnostic_level_not_selectable" not in assessment.loc[
        assessment["employee_id"] == 2, "violation_reason_if_none"
    ].iloc[0]
    assert assessment["final_group"].notna().sum() == 3


def test_skipped_diagnostic_levels_do_not_imply_no_recourse():
    cohort = pd.DataFrame([{"EmployeeNumber": 1, "attrition_score": .8}])
    statuses = pd.DataFrame([
        {"employee_id": 1, "constraint_level": 1, "success": False,
         "error": "skipped: DiCE disabled by config"},
        {"employee_id": 1, "constraint_level": 2, "success": False,
         "error": "skipped: DiCE disabled by config"},
        {"employee_id": 1, "constraint_level": 3, "success": True, "error": ""},
    ])
    assessment, _ = build_assessment(
        cohort, .5, pd.DataFrame(columns=SCENARIO_COLUMNS), statuses,
    )
    row = assessment.iloc[0]
    assert row["analysis_status"] == "partial"
    assert pd.isna(row["final_group"])
    assert row["violation_reason_if_none"].startswith("analysis_incomplete:")
