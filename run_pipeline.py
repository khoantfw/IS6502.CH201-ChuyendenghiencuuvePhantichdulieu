#!/usr/bin/env python3
"""Run the complete HR intervention feasibility research pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hr_recourse.counterfactuals import (  # noqa: E402
    CounterfactualRun, empty_scenarios, run_dice_level, run_level3_for_cohort,
    salary_cap,
)
from hr_recourse.data_modeling import (  # noqa: E402
    ID_COLUMN, TARGET, choose_threshold, classification_metrics, feature_columns,
    fit_and_compare_models, load_and_validate_data, predict_scores, split_data,
    split_summary,
)
from hr_recourse.explainability import compute_shap_importance  # noqa: E402
from hr_recourse.feasibility import (  # noqa: E402
    audit_diagnostic_scenarios, build_assessment, found_rate_by_level,
    violation_summary,
)
from hr_recourse.reporting import (  # noqa: E402
    create_eda_outputs, ensure_output_dirs, feature_grouping_table, plot_assessment,
    plot_importance, plot_model_outputs, plot_supporting_analyses, write_csv,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _skipped_status(cohort: pd.DataFrame, level: int, reason: str) -> pd.DataFrame:
    """Record an intentionally skipped level without treating it as completed."""
    return pd.DataFrame([{"employee_id": row[ID_COLUMN], "constraint_level": level,
                          "success": False, "error": f"skipped: {reason}"}
                         for _, row in cohort.iterrows()])


def _concat_scenarios(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else empty_scenarios()


def _cleanup_disabled_outputs(config: dict[str, Any], paths: dict[str, Path]) -> None:
    """Remove generated artifacts that would otherwise be stale after a partial run."""
    run_cfg = config["run"]
    stale: list[Path] = []
    if not run_cfg["run_shap"]:
        stale.extend([
            paths["tables"] / "shap_feature_importance.csv",
            paths["tables"] / "shap_group_importance.csv",
            paths["tables"] / "case_study_details.csv",
            paths["figures"] / "shap_feature_importance.png",
            paths["figures"] / "shap_group_importance.png",
        ])
    if not run_cfg["run_sensitivity"]:
        stale.extend([
            paths["tables"] / "sensitivity_threshold.csv",
            paths["tables"] / "sensitivity_salary_cap.csv",
            paths["tables"] / "sensitivity_overtime.csv",
            paths["figures"] / "sensitivity_threshold.png",
            paths["figures"] / "sensitivity_salary_cap.png",
            paths["figures"] / "sensitivity_overtime.png",
        ])
    if not run_cfg["run_stability"]:
        stale.extend([
            paths["tables"] / "stability_by_seed.csv",
            paths["figures"] / "stability_by_seed.png",
        ])
    if not run_cfg["generate_figures"]:
        stale.extend(paths["figures"].glob("*.png"))
    for path in set(stale):
        path.unlink(missing_ok=True)


def analyze_cohort(
    train: pd.DataFrame, cohort: pd.DataFrame, model: Any, threshold: float,
    config: dict[str, Any], run_dice: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cf = config["counterfactuals"]
    if cohort.empty:
        status = pd.DataFrame(columns=["employee_id", "constraint_level", "success", "error"])
        assessment, best = build_assessment(cohort, threshold, empty_scenarios(), status)
        return empty_scenarios(), empty_scenarios(), empty_scenarios(), assessment, status
    if run_dice:
        level1 = run_dice_level(
            train, cohort, model, threshold, 1, cf["dice_total"], cf["retry_count"],
            config["primary_seed"], cf["salary_cap_percentile"], cf["minimum_salary_group_size"]
        )
        level2 = run_dice_level(
            train, cohort, model, threshold, 2, cf["dice_total"], cf["retry_count"],
            config["primary_seed"], cf["salary_cap_percentile"], cf["minimum_salary_group_size"]
        )
    else:
        level1 = CounterfactualRun(empty_scenarios(), _skipped_status(cohort, 1, "DiCE disabled by config"))
        level2 = CounterfactualRun(empty_scenarios(), _skipped_status(cohort, 2, "DiCE disabled by config"))
    level3 = run_level3_for_cohort(
        cohort, train, model, threshold, cf["salary_increments"],
        cf["salary_cap_percentile"], cf["minimum_salary_group_size"], cf["overtime_policy"],
    )
    l1_audited = audit_diagnostic_scenarios(level1.scenarios)
    l2_audited = audit_diagnostic_scenarios(level2.scenarios)
    all_scenarios = _concat_scenarios([l1_audited, l2_audited, level3.scenarios])
    statuses = pd.concat([level1.status, level2.status, level3.status], ignore_index=True)
    assessment, best = build_assessment(cohort, threshold, all_scenarios, statuses)
    return l1_audited, l2_audited, level3.scenarios, assessment, statuses


def fit_seed(df: pd.DataFrame, config: dict[str, Any], seed: int) -> dict[str, Any]:
    splits = split_data(df, seed, config["train_size"], config["validation_size"], config["test_size"])
    winner, comparison = fit_and_compare_models(splits.train, splits.validation, seed)
    threshold_cfg = config["threshold"]
    threshold, threshold_table = choose_threshold(
        splits.validation[TARGET], winner.validation_scores,
        threshold_cfg["minimum"], threshold_cfg["maximum"], threshold_cfg["step"],
    )
    test_scores = predict_scores(winner.pipeline, splits.test)
    metrics, cm = classification_metrics(splits.test[TARGET], test_scores, threshold, "test")
    test_scored = splits.test.copy()
    test_scored["attrition_score"] = test_scores
    cohort = test_scored[test_scored["attrition_score"] >= threshold].copy()
    return {"splits": splits, "winner": winner, "comparison": comparison,
            "threshold": threshold, "threshold_table": threshold_table,
            "test_scores": test_scores, "metrics": metrics, "cm": cm, "cohort": cohort}


def _select_case_studies(assessment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = assessment[assessment["analysis_status"] == "completed"]
    for group, frame in completed.groupby("final_group"):
        median = frame["attrition_score"].median()
        selected = frame.loc[(frame["attrition_score"] - median).abs().idxmin()].copy()
        selected["selection_rule"] = "closest_to_group_median_attrition_score"
        rows.append(selected)
    return pd.DataFrame(rows)


def write_supporting_tables(
    assessment: pd.DataFrame, best: pd.DataFrame, cohort: pd.DataFrame,
    all_scenarios: pd.DataFrame, paths: dict[str, Path],
) -> None:
    completed = assessment[assessment["analysis_status"] == "completed"]
    group_summary = completed.groupby("final_group", dropna=False).agg(
        n_employees=("employee_id", "size"),
        mean_attrition_score=("attrition_score", "mean"),
        median_attrition_score=("attrition_score", "median"),
    ).reset_index()
    group_summary["share"] = group_summary["n_employees"] / max(len(completed), 1)
    write_csv(group_summary, paths["tables"] / "final_group_summary.csv")
    all_feasible = all_scenarios[
        (all_scenarios["constraint_level"] == 3)
        & all_scenarios["model_valid"].astype(bool)
        & all_scenarios["hr_feasible"].astype(bool)
    ]
    lever_rows = []
    for scope, frame in (("all feasible Level 3", all_feasible), ("selected best", best)):
        for lever in ("MonthlyIncome", "StockOptionLevel", "OverTime"):
            count = frame["changed_features"].fillna("").str.split("|").apply(lambda values: lever in values).sum() if not frame.empty else 0
            lever_rows.append({
                "scope": scope, "hr_lever": lever, "n_scenarios": len(frame),
                "n_scenarios_using_lever": int(count),
                "share_of_scenarios": count / len(frame) if len(frame) else np.nan,
            })
    write_csv(pd.DataFrame(lever_rows), paths["tables"] / "hr_lever_contribution.csv")

    merged = cohort.merge(assessment[["employee_id", "final_group", "analysis_status"]],
                          left_on=ID_COLUMN, right_on="employee_id", how="inner")
    characteristics = merged.groupby("final_group", dropna=False).agg(
        n_employees=("employee_id", "size"),
        actual_attrition_rate=(TARGET, "mean"),
        mean_attrition_score=("attrition_score", "mean"),
        median_attrition_score=("attrition_score", "median"),
        overtime_yes_share=("OverTime", lambda values: values.eq("Yes").mean()),
        median_monthly_income=("MonthlyIncome", "median"),
        median_stock_option_level=("StockOptionLevel", "median"),
        median_job_level=("JobLevel", "median"),
        median_years_at_company=("YearsAtCompany", "median"),
    ).reset_index()
    write_csv(characteristics, paths["tables"] / "actionability_group_characteristics.csv")

    role_counts = merged.groupby(["JobRole", "final_group"]).size().unstack(fill_value=0)
    role_counts["n_high_risk"] = role_counts.sum(axis=1)
    if "actionable" not in role_counts:
        role_counts["actionable"] = 0
    role_counts["actionable_rate"] = role_counts["actionable"] / role_counts["n_high_risk"]
    write_csv(role_counts.reset_index(), paths["tables"] / "actionability_by_job_role.csv")


def write_detailed_case_studies(
    assessment: pd.DataFrame, cohort: pd.DataFrame, train: pd.DataFrame,
    local_shap: pd.DataFrame, config: dict[str, Any], paths: dict[str, Path],
) -> None:
    selected = _select_case_studies(assessment)
    rows = []
    cap_cfg = config["counterfactuals"]
    for _, case in selected.iterrows():
        employee = cohort[cohort[ID_COLUMN] == case["employee_id"]].iloc[0]
        shap_row = local_shap[local_shap["employee_id"] == case["employee_id"]].iloc[0].drop("employee_id")
        top_features = shap_row.abs().sort_values(ascending=False).head(5).index
        top_shap = "|".join(f"{feature}:{shap_row[feature]:+.4f}" for feature in top_features)
        increase = float(case["salary_increase_pct"]) if pd.notna(case["salary_increase_pct"]) else np.nan
        stock_increment = int(case["stock_option_increment"]) if pd.notna(case["stock_option_increment"]) else np.nan
        rows.append({
            "employee_id": case["employee_id"], "final_group": case["final_group"],
            "actual_attrition": int(employee[TARGET]), "score_before": case["attrition_score"],
            "threshold": case["high_risk_threshold"], "best_scenario_id": case["best_scenario_id"],
            "changed_features": case["best_changed_features"], "score_after": case["score_after_best_scenario"],
            "monthly_income_before": employee["MonthlyIncome"],
            "monthly_income_after": employee["MonthlyIncome"] * (1 + increase / 100) if pd.notna(increase) else np.nan,
            "salary_cap": salary_cap(train, employee, cap_cfg["salary_cap_percentile"], cap_cfg["minimum_salary_group_size"]),
            "stock_option_before": employee["StockOptionLevel"],
            "stock_option_after": employee["StockOptionLevel"] + stock_increment if pd.notna(stock_increment) else np.nan,
            "overtime_before": employee["OverTime"],
            "overtime_after": ("No" if case["overtime_change"] == "Yes->No" else employee["OverTime"]) if pd.notna(case["overtime_change"]) else np.nan,
            "top_local_shap_features": top_shap,
            "reason_if_no_feasible_scenario": case["violation_reason_if_none"],
            "selection_rule": case["selection_rule"],
        })
    write_csv(pd.DataFrame(rows), paths["tables"] / "case_study_details.csv")


def _level3_summary(cohort: pd.DataFrame, scenarios: pd.DataFrame, label: str, value: Any) -> dict[str, Any]:
    feasible_ids = scenarios.loc[scenarios["hr_feasible"].astype(bool), "employee_id"].nunique() if not scenarios.empty else 0
    return {"assumption": label, "value": value, "n_high_risk": len(cohort),
            "n_actionable": feasible_ids, "feasibility_rate": feasible_ids / len(cohort) if len(cohort) else np.nan}


def run_sensitivity(primary: dict[str, Any], config: dict[str, Any], paths: dict[str, Path]) -> None:
    splits, model = primary["splits"], primary["winner"].pipeline
    test_scored = splits.test.copy()
    test_scored["attrition_score"] = primary["test_scores"]
    cf = config["counterfactuals"]
    threshold_rows = []
    seen = set()
    for offset in config["sensitivity"]["threshold_offsets"]:
        threshold = float(np.clip(primary["threshold"] + offset, 0.05, 0.95))
        if threshold in seen:
            continue
        seen.add(threshold)
        cohort = test_scored[test_scored["attrition_score"] >= threshold].copy()
        _, _, level3_scenarios, assessment, _ = analyze_cohort(
            splits.train, cohort, model, threshold, config, run_dice=config["run"]["run_dice"]
        )
        row = _level3_summary(cohort, level3_scenarios, "threshold", threshold)
        completed = assessment[assessment["analysis_status"] == "completed"]
        counts = completed["final_group"].value_counts()
        row.update({
            "n_completed": len(completed), "n_failed": len(assessment) - len(completed),
            "actionable_rate": counts.get("actionable", 0) / len(completed) if len(completed) else np.nan,
            "model_valid_not_hr_feasible_rate": counts.get("model_valid_not_hr_feasible", 0) / len(completed) if len(completed) else np.nan,
            "no_model_recourse_within_search_rate": counts.get("no_model_recourse_within_search", 0) / len(completed) if len(completed) else np.nan,
        })
        threshold_rows.append(row)
    write_csv(pd.DataFrame(threshold_rows), paths["tables"] / "sensitivity_threshold.csv")

    cohort = primary["cohort"]
    cap_rows = []
    for percentile in config["sensitivity"]["salary_cap_percentiles"]:
        level3 = run_level3_for_cohort(cohort, splits.train, model, primary["threshold"], cf["salary_increments"],
                                       percentile, cf["minimum_salary_group_size"], cf["overtime_policy"])
        cap_rows.append(_level3_summary(cohort, level3.scenarios, "salary_cap_percentile", percentile))
    write_csv(pd.DataFrame(cap_rows), paths["tables"] / "sensitivity_salary_cap.csv")

    overtime_rows = []
    for policy in config["sensitivity"]["overtime_policies"]:
        level3 = run_level3_for_cohort(cohort, splits.train, model, primary["threshold"], cf["salary_increments"],
                                       cf["salary_cap_percentile"], cf["minimum_salary_group_size"], policy)
        overtime_rows.append(_level3_summary(cohort, level3.scenarios, "overtime_policy", policy))
    write_csv(pd.DataFrame(overtime_rows), paths["tables"] / "sensitivity_overtime.csv")


def run_stability(df: pd.DataFrame, config: dict[str, Any], primary: dict[str, Any], paths: dict[str, Path]) -> None:
    rows = []
    seeds = [config["primary_seed"], *config["stability_seeds"]]
    for seed in seeds:
        try:
            fitted = primary if seed == config["primary_seed"] else fit_seed(df, config, seed)
            _, _, level3, assessment, _ = analyze_cohort(
                fitted["splits"].train, fitted["cohort"], fitted["winner"].pipeline,
                fitted["threshold"], {**config, "primary_seed": seed}, run_dice=config["run"]["run_dice"],
            )
            completed = assessment[assessment["analysis_status"] == "completed"]
            counts = completed["final_group"].value_counts()
            rows.append({
                "seed": seed, "status": "completed", "selected_model": fitted["winner"].name,
                "test_average_precision": float(fitted["metrics"].iloc[0]["average_precision"]),
                "threshold": fitted["threshold"], "n_high_risk": len(fitted["cohort"]),
                "n_completed": len(completed), "n_failed": len(assessment) - len(completed),
                "feasibility_rate": float(completed["level3_feasible_scenario_found"].mean()) if len(completed) else np.nan,
                "actionable_rate": counts.get("actionable", 0) / len(completed) if len(completed) else np.nan,
                "model_valid_not_hr_feasible_rate": counts.get("model_valid_not_hr_feasible", 0) / len(completed) if len(completed) else np.nan,
                "no_model_recourse_within_search_rate": counts.get("no_model_recourse_within_search", 0) / len(completed) if len(completed) else np.nan,
            })
        except Exception as exc:
            rows.append({"seed": seed, "status": "failed", "error": str(exc)})
    write_csv(pd.DataFrame(rows), paths["tables"] / "stability_by_seed.csv")


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    data_path = ROOT / config["data_path"]
    paths = ensure_output_dirs(ROOT / config["output_dir"])
    _cleanup_disabled_outputs(config, paths)
    df, quality = load_and_validate_data(data_path)
    write_csv(quality, paths["tables"] / "data_quality_summary.csv")
    write_csv(feature_grouping_table(), paths["tables"] / "feature_grouping.csv")
    if config["run"]["generate_figures"]:
        create_eda_outputs(df, paths["tables"], paths["figures"])

    primary = fit_seed(df, config, config["primary_seed"])
    write_csv(split_summary(primary["splits"]), paths["tables"] / "split_summary.csv")
    write_csv(primary["comparison"], paths["tables"] / "model_comparison.csv")
    write_csv(primary["threshold_table"], paths["tables"] / "validation_threshold_search.csv")
    write_csv(primary["metrics"], paths["tables"] / "final_model_metrics.csv")
    high_risk_columns = [ID_COLUMN, TARGET, "attrition_score"]
    write_csv(primary["cohort"][high_risk_columns], paths["tables"] / "high_risk_employees.csv")
    joblib.dump({"model": primary["winner"].pipeline, "threshold": primary["threshold"],
                 "model_name": primary["winner"].name, "features": feature_columns(df)},
                paths["models"] / "final_model.joblib")
    if config["run"]["generate_figures"]:
        plot_model_outputs(primary["splits"].test[TARGET], primary["test_scores"], primary["threshold"], primary["cm"], paths["figures"])

    l1, l2, l3, assessment, statuses = analyze_cohort(
        primary["splits"].train, primary["cohort"], primary["winner"].pipeline,
        primary["threshold"], config, run_dice=config["run"]["run_dice"],
    )
    all_scenarios = _concat_scenarios([l1, l2, l3])
    _, best = build_assessment(primary["cohort"], primary["threshold"], all_scenarios, statuses)
    write_csv(l1, paths["scenarios"] / "dice_level_1.csv")
    write_csv(l2, paths["scenarios"] / "dice_level_2.csv")
    write_csv(l3, paths["scenarios"] / "level_3_grid.csv")
    write_csv(statuses, paths["tables"] / "counterfactual_analysis_status.csv")
    write_csv(found_rate_by_level(all_scenarios, primary["cohort"], statuses), paths["tables"] / "counterfactual_found_rate_by_level.csv")
    write_csv(assessment, paths["tables"] / "hr_intervention_feasibility_assessment.csv")
    write_csv(best, paths["tables"] / "best_feasible_scenarios.csv")
    write_csv(violation_summary(all_scenarios), paths["tables"] / "violation_summary.csv")
    write_supporting_tables(assessment, best, primary["cohort"], all_scenarios, paths)
    if config["run"]["generate_figures"]:
        plot_assessment(assessment, paths["figures"])

    if config["run"]["run_shap"]:
        feature_imp, group_imp, local = compute_shap_importance(
            primary["winner"].pipeline, primary["splits"].train, primary["splits"].test,
            config["primary_seed"],
        )
        write_csv(feature_imp, paths["tables"] / "shap_feature_importance.csv")
        write_csv(group_imp, paths["tables"] / "shap_group_importance.csv")
        write_detailed_case_studies(
            assessment, primary["cohort"], primary["splits"].train,
            local, config, paths,
        )
        if config["run"]["generate_figures"]:
            plot_importance(feature_imp, group_imp, paths["figures"])

    if config["run"]["run_sensitivity"]:
        run_sensitivity(primary, config, paths)
    if config["run"]["run_stability"]:
        run_stability(df, config, primary, paths)

    supporting_files = {
        "found_rate": paths["tables"] / "counterfactual_found_rate_by_level.csv",
        "violations": paths["tables"] / "violation_summary.csv",
        "levers": paths["tables"] / "hr_lever_contribution.csv",
        "threshold": paths["tables"] / "sensitivity_threshold.csv",
        "cap": paths["tables"] / "sensitivity_salary_cap.csv",
        "overtime": paths["tables"] / "sensitivity_overtime.csv",
        "stability": paths["tables"] / "stability_by_seed.csv",
    }
    if (
        config["run"]["generate_figures"]
        and config["run"]["run_sensitivity"]
        and config["run"]["run_stability"]
        and all(path.exists() for path in supporting_files.values())
    ):
        plot_supporting_analyses(
            pd.read_csv(supporting_files["found_rate"]),
            pd.read_csv(supporting_files["violations"]),
            pd.read_csv(supporting_files["levers"]),
            pd.read_csv(supporting_files["threshold"]),
            pd.read_csv(supporting_files["cap"]),
            pd.read_csv(supporting_files["overtime"]),
            pd.read_csv(supporting_files["stability"]),
            paths["figures"],
        )

    summary = {
        "selected_model": primary["winner"].name,
        "high_risk_threshold": primary["threshold"],
        "n_high_risk": len(primary["cohort"]),
        "assessment_status_counts": assessment["analysis_status"].value_counts(dropna=False).to_dict(),
        "final_group_counts": assessment["final_group"].value_counts(dropna=False).to_dict(),
    }
    (ROOT / config["output_dir"] / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    run(args.config)


if __name__ == "__main__":
    main()
