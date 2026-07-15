"""Feasibility audit, Level 3 ranking, and employee assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_modeling import FEATURE_GROUPS, ID_COLUMN

ASSESSMENT_COLUMNS = [
    "employee_id", "predicted_risk", "attrition_score", "high_risk_threshold",
    "model_valid_scenario_found", "level3_feasible_scenario_found", "best_scenario_id",
    "best_changed_features", "score_after_best_scenario", "salary_increase_pct",
    "stock_option_increment", "overtime_change", "violation_reason_if_none",
    "final_group", "analysis_status",
]


def audit_diagnostic_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Attach transparent group-level violations to Level 1/2 scenarios."""
    if scenarios.empty:
        return scenarios.copy()
    audited = scenarios.copy()
    for idx, row in audited.iterrows():
        changed = set(filter(None, str(row["changed_features"]).split("|")))
        violations = [code for code in str(row.get("violations", "")).split("|")
                      if code and code != "pending_audit"]
        if changed.intersection(FEATURE_GROUPS["immutable"]):
            violations.append("immutable_changed")
        if changed.intersection(FEATURE_GROUPS["historical"]):
            violations.append("historical_changed")
        if changed.intersection(FEATURE_GROUPS["perception_experience"]):
            violations.append("perception_as_direct_action")
        if changed.intersection(FEATURE_GROUPS["job_structure_context"]):
            violations.append("job_structure_outside_main_scope")
        if row["constraint_level"] in (1, 2):
            violations.append("diagnostic_level_not_selectable")
        audited.at[idx, "violations"] = "|".join(dict.fromkeys(violations))
        audited.at[idx, "hr_feasible"] = False
    return audited


def rank_level3(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return valid and feasible Level 3 scenarios in the pre-registered order."""
    if scenarios.empty:
        return scenarios.copy()
    eligible = scenarios[
        (scenarios["constraint_level"] == 3)
        & scenarios["model_valid"].astype(bool)
        & scenarios["hr_feasible"].astype(bool)
    ].copy()
    if eligible.empty:
        return eligible
    eligible["has_salary_increase"] = eligible["salary_increase_pct"].gt(1e-7)
    eligible["score_reduction"] = eligible["score_before"] - eligible["score_after"]
    eligible = eligible.sort_values(
        ["employee_id", "n_changes", "has_salary_increase", "stock_option_increment",
         "salary_increase_pct", "score_reduction", "scenario_id"],
        ascending=[True, True, True, True, True, False, True],
    )
    eligible["rank"] = eligible.groupby("employee_id").cumcount() + 1
    return eligible


def build_assessment(
    cohort: pd.DataFrame, threshold: float, all_scenarios: pd.DataFrame,
    statuses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign one outcome group only when all evidence required for it exists."""
    ranked = rank_level3(all_scenarios)
    best = ranked[ranked["rank"] == 1].copy() if not ranked.empty else ranked.copy()
    rows = []
    for _, employee in cohort.iterrows():
        employee_id = employee[ID_COLUMN]
        employee_scenarios = all_scenarios[all_scenarios["employee_id"] == employee_id]
        employee_status = statuses[statuses["employee_id"] == employee_id]
        completed_levels = set(employee_status.loc[employee_status["success"].astype(bool), "constraint_level"].astype(int))
        incomplete = completed_levels != {1, 2, 3}
        errors = employee_status.loc[~employee_status["success"].astype(bool), "error"].dropna().astype(str).tolist()
        skipped = any(error.startswith("skipped:") for error in errors)
        any_valid = bool(employee_scenarios["model_valid"].astype(bool).any()) if not employee_scenarios.empty else False
        best_row = best[best["employee_id"] == employee_id]
        has_feasible = not best_row.empty

        if has_feasible:
            analysis_status, final_group = "completed", "actionable"
        elif skipped:
            analysis_status, final_group = "partial", np.nan
        elif incomplete:
            analysis_status, final_group = "failed", np.nan
        elif any_valid:
            analysis_status, final_group = "completed", "model_valid_not_hr_feasible"
        else:
            analysis_status, final_group = "completed", "no_model_recourse_within_search"

        selected = best_row.iloc[0] if has_feasible else None
        if skipped:
            reason = "analysis_incomplete: " + " | ".join(errors)
        elif incomplete:
            reason = "technical_failure: " + " | ".join(errors)
        elif not any_valid:
            reason = "no_model_valid_scenario_within_search"
        elif not has_feasible:
            violations = employee_scenarios.loc[employee_scenarios["model_valid"].astype(bool), "violations"]
            reason_codes = {
                v for cell in violations.astype(str) for v in cell.split("|")
                if v and v != "diagnostic_level_not_selectable"
            }
            reason = "|".join(sorted(reason_codes)) or "no_level3_feasible_scenario"
        else:
            reason = ""
        rows.append({
            "employee_id": employee_id, "predicted_risk": "high",
            "attrition_score": float(employee["attrition_score"]),
            "high_risk_threshold": threshold,
            "model_valid_scenario_found": any_valid,
            "level3_feasible_scenario_found": has_feasible,
            "best_scenario_id": selected["scenario_id"] if selected is not None else np.nan,
            "best_changed_features": selected["changed_features"] if selected is not None else np.nan,
            "score_after_best_scenario": selected["score_after"] if selected is not None else np.nan,
            "salary_increase_pct": selected["salary_increase_pct"] if selected is not None else np.nan,
            "stock_option_increment": selected["stock_option_increment"] if selected is not None else np.nan,
            "overtime_change": selected["overtime_change"] if selected is not None else np.nan,
            "violation_reason_if_none": reason, "final_group": final_group,
            "analysis_status": analysis_status,
        })
    return pd.DataFrame(rows, columns=ASSESSMENT_COLUMNS), best.drop(columns=["has_salary_increase", "score_reduction", "rank"], errors="ignore")


def violation_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    valid = scenarios[scenarios["model_valid"].astype(bool)] if not scenarios.empty else scenarios
    values = [code for cell in valid.get("violations", pd.Series(dtype=str)).fillna("").astype(str)
              for code in cell.split("|") if code and code != "diagnostic_level_not_selectable"]
    if not values:
        return pd.DataFrame(columns=["violation", "count", "share_of_violations"])
    counts = pd.Series(values).value_counts().rename_axis("violation").reset_index(name="count")
    counts["share_of_violations"] = counts["count"] / counts["count"].sum()
    return counts


def found_rate_by_level(scenarios: pd.DataFrame, cohort: pd.DataFrame, statuses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    denominator = len(cohort)
    for level in (1, 2, 3):
        level_rows = scenarios[scenarios["constraint_level"] == level]
        found = level_rows.loc[level_rows["model_valid"].astype(bool), "employee_id"].nunique()
        successful = statuses[(statuses["constraint_level"] == level) & statuses["success"].astype(bool)]["employee_id"].nunique()
        rows.append({"constraint_level": level, "n_high_risk": denominator,
                     "n_successfully_analyzed": successful, "n_with_model_valid_scenario": found,
                     "found_rate": found / denominator if denominator else np.nan})
    return pd.DataFrame(rows)
