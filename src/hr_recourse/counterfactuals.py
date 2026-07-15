"""Counterfactual generation for broad DiCE spaces and the main HR grid."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .data_modeling import FEATURE_GROUPS, ID_COLUMN, TARGET, feature_columns

SCENARIO_COLUMNS = [
    "employee_id", "constraint_level", "scenario_id", "score_before", "score_after",
    "changed_features", "changes_json", "n_changes", "salary_increase_pct", "stock_option_increment",
    "overtime_change", "model_valid", "hr_feasible", "violations",
]


@dataclass
class CounterfactualRun:
    scenarios: pd.DataFrame
    status: pd.DataFrame


class ThresholdModelAdapter(BaseEstimator, ClassifierMixin):
    """Move a model's research threshold to DiCE's conventional 0.5 boundary."""

    def __init__(self, model: Any, threshold: float):
        self.model = model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ThresholdModelAdapter":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.model.predict_proba(X)[:, 1], 1e-8, 1 - 1e-8)
        threshold = np.clip(self.threshold, 1e-8, 1 - 1e-8)
        adjusted_logit = np.log(raw / (1 - raw)) - np.log(threshold / (1 - threshold))
        adjusted = 1 / (1 + np.exp(-adjusted_logit))
        return np.column_stack([1 - adjusted, adjusted])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def empty_scenarios() -> pd.DataFrame:
    return pd.DataFrame(columns=SCENARIO_COLUMNS)


def is_no_counterfactual_result(error: str) -> bool:
    """Return True when DiCE completed its search but found no valid result."""
    return "No counterfactuals found" in error


def _changed_features(original: pd.Series, candidate: pd.Series, features: Iterable[str]) -> list[str]:
    changed = []
    for feature in features:
        before, after = original[feature], candidate[feature]
        if pd.isna(before) and pd.isna(after):
            continue
        if isinstance(before, (float, np.floating)) or isinstance(after, (float, np.floating)):
            try:
                if not np.isclose(float(before), float(after), rtol=1e-7, atol=1e-7):
                    changed.append(feature)
                continue
            except (TypeError, ValueError):
                pass
        if before != after:
            changed.append(feature)
    return changed


def _json_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values into strict JSON-compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _changes_json(original: pd.Series, candidate: pd.Series, changed: Iterable[str]) -> str:
    """Preserve before/after values so exported scenarios can be independently audited."""
    payload = {
        feature: {
            "before": _json_scalar(original[feature]),
            "after": _json_scalar(candidate[feature]),
        }
        for feature in changed
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def salary_cap(train: pd.DataFrame, employee: pd.Series, percentile: float = 75,
               minimum_group_size: int = 10) -> float:
    role_level = train[(train["JobRole"] == employee["JobRole"]) & (train["JobLevel"] == employee["JobLevel"])]
    if len(role_level) >= minimum_group_size:
        reference = role_level
    else:
        level = train[train["JobLevel"] == employee["JobLevel"]]
        reference = level if len(level) >= minimum_group_size else train
    cap = float(np.percentile(reference["MonthlyIncome"], percentile))
    return max(float(employee["MonthlyIncome"]), cap)


def generate_level3_candidates(
    employee: pd.Series, train: pd.DataFrame, salary_increments: Iterable[float],
    salary_cap_percentile: float = 75, minimum_group_size: int = 10,
    overtime_policy: str = "allowed",
) -> tuple[pd.DataFrame, float]:
    """Exhaust the explicitly allowed Level 3 space, excluding the original profile."""
    cap = salary_cap(train, employee, salary_cap_percentile, minimum_group_size)
    income = float(employee["MonthlyIncome"])
    salaries = sorted({round(min(income * (1 + float(pct)), cap), 2) for pct in salary_increments})
    max_stock = int(train["StockOptionLevel"].max())
    stocks = list(range(int(employee["StockOptionLevel"]), max_stock + 1))
    overtime = [employee["OverTime"]]
    if overtime_policy == "allowed" and employee["OverTime"] == "Yes":
        overtime.append("No")
    if overtime_policy not in {"allowed", "locked"}:
        raise ValueError("overtime_policy must be 'allowed' or 'locked'")

    rows = []
    seen: set[tuple[float, int, str]] = set()
    for new_income in salaries:
        for new_stock in stocks:
            for new_overtime in overtime:
                key = (new_income, new_stock, new_overtime)
                if key in seen:
                    continue
                seen.add(key)
                if np.isclose(new_income, income) and new_stock == employee["StockOptionLevel"] and new_overtime == employee["OverTime"]:
                    continue
                row = employee.copy()
                row["MonthlyIncome"] = new_income
                row["StockOptionLevel"] = new_stock
                row["OverTime"] = new_overtime
                rows.append(row)
    return pd.DataFrame(rows), cap


def evaluate_level3(
    employee: pd.Series, train: pd.DataFrame, model: Any, threshold: float,
    salary_increments: Iterable[float], salary_cap_percentile: float = 75,
    minimum_group_size: int = 10, overtime_policy: str = "allowed",
) -> pd.DataFrame:
    candidates, cap = generate_level3_candidates(
        employee, train, salary_increments, salary_cap_percentile,
        minimum_group_size, overtime_policy,
    )
    if candidates.empty:
        return empty_scenarios()
    features = feature_columns(train)
    before = float(model.predict_proba(pd.DataFrame([employee])[features])[:, 1][0])
    after_scores = model.predict_proba(candidates[features])[:, 1]
    result = []
    for sequence, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        changed = _changed_features(employee, candidate, features)
        income_before, income_after = float(employee["MonthlyIncome"]), float(candidate["MonthlyIncome"])
        violations = []
        if income_after < income_before - 1e-7:
            violations.append("salary_decreased")
        if income_after > cap + 1e-7:
            violations.append("salary_cap_exceeded")
        if income_after > income_before * 1.30 + 0.01:
            violations.append("salary_increase_over_30pct")
        if candidate["StockOptionLevel"] < employee["StockOptionLevel"]:
            violations.append("stock_option_decreased")
        if candidate["StockOptionLevel"] > train["StockOptionLevel"].max():
            violations.append("stock_option_out_of_range")
        if employee["OverTime"] == "No" and candidate["OverTime"] != "No":
            violations.append("overtime_wrong_direction")
        disallowed = set(changed).difference(FEATURE_GROUPS["directly_actionable"])
        if disallowed:
            violations.append("changed_outside_level3")
        score_after = float(after_scores[sequence - 1])
        model_valid = score_after < threshold
        result.append({
            "employee_id": employee[ID_COLUMN], "constraint_level": 3,
            "scenario_id": f"{employee[ID_COLUMN]}-L3-{sequence:03d}",
            "score_before": before, "score_after": score_after,
            "changed_features": "|".join(changed),
            "changes_json": _changes_json(employee, candidate, changed),
            "n_changes": len(changed),
            "salary_increase_pct": (income_after / income_before - 1) * 100,
            "stock_option_increment": int(candidate["StockOptionLevel"] - employee["StockOptionLevel"]),
            "overtime_change": f"{employee['OverTime']}->{candidate['OverTime']}" if candidate["OverTime"] != employee["OverTime"] else "none",
            "model_valid": bool(model_valid), "hr_feasible": bool(model_valid and not violations),
            "violations": "|".join(violations),
        })
    return pd.DataFrame(result, columns=SCENARIO_COLUMNS)


def run_level3_for_cohort(
    cohort: pd.DataFrame, train: pd.DataFrame, model: Any, threshold: float,
    salary_increments: Iterable[float], salary_cap_percentile: float = 75,
    minimum_group_size: int = 10, overtime_policy: str = "allowed",
) -> CounterfactualRun:
    scenario_frames, statuses = [], []
    for _, employee in cohort.iterrows():
        try:
            scenario_frames.append(evaluate_level3(
                employee, train, model, threshold, salary_increments,
                salary_cap_percentile, minimum_group_size, overtime_policy,
            ))
            statuses.append({"employee_id": employee[ID_COLUMN], "constraint_level": 3, "success": True, "error": ""})
        except Exception as exc:  # employee-level isolation is intentional
            statuses.append({"employee_id": employee[ID_COLUMN], "constraint_level": 3, "success": False, "error": str(exc)})
    nonempty = [frame for frame in scenario_frames if not frame.empty]
    scenarios = pd.concat(nonempty, ignore_index=True) if nonempty else empty_scenarios()
    return CounterfactualRun(scenarios, pd.DataFrame(statuses))


def _dice_permitted_ranges(train_features: pd.DataFrame) -> dict[str, Any]:
    ranges: dict[str, Any] = {}
    for column in train_features.columns:
        if train_features[column].dtype == "object":
            ranges[column] = train_features[column].dropna().unique().tolist()
        else:
            ranges[column] = [float(train_features[column].min()), float(train_features[column].max())]
    return ranges


def _diagnostic_constraint_violations(
    original: pd.Series, candidate: pd.Series, train: pd.DataFrame,
    cap_percentile: float, minimum_group_size: int,
) -> list[str]:
    violations: list[str] = []
    cap = salary_cap(train, original, cap_percentile, minimum_group_size)
    if float(candidate["MonthlyIncome"]) < float(original["MonthlyIncome"]) - 1e-7:
        violations.append("salary_decreased")
    if float(candidate["MonthlyIncome"]) > cap + 1e-7:
        violations.append("salary_cap_exceeded")
    if candidate["StockOptionLevel"] < original["StockOptionLevel"]:
        violations.append("stock_option_decreased")
    if candidate["StockOptionLevel"] > train["StockOptionLevel"].max():
        violations.append("stock_option_out_of_range")
    if candidate["OverTime"] != original["OverTime"] and not (
        original["OverTime"] == "Yes" and candidate["OverTime"] == "No"
    ):
        violations.append("overtime_wrong_direction")
    for years_field in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
        if candidate[years_field] > candidate["YearsAtCompany"]:
            violations.append("invalid_tenure_relationship")
            break
    for column in feature_columns(train):
        value = candidate[column]
        if train[column].dtype == "object":
            invalid = value not in set(train[column].dropna())
        else:
            invalid = value < train[column].min() or value > train[column].max()
        if invalid:
            violations.append("value_outside_train_domain")
            break
    return sorted(set(violations))


def run_dice_level(
    train: pd.DataFrame, cohort: pd.DataFrame, model: Any, threshold: float, level: int,
    total_cfs: int = 5, retry_count: int = 1, seed: int = 42,
    salary_cap_percentile: float = 75, minimum_group_size: int = 10,
) -> CounterfactualRun:
    """Generate Level 1/2 DiCE scenarios and re-score them with the raw model."""
    if level not in {1, 2}:
        raise ValueError("DiCE is used only for levels 1 and 2")
    try:
        import dice_ml
    except ImportError as exc:
        error = "dice-ml is not installed; run pip install -r requirements.txt"
        statuses = [{"employee_id": row[ID_COLUMN], "constraint_level": level, "success": False, "error": error}
                    for _, row in cohort.iterrows()]
        return CounterfactualRun(empty_scenarios(), pd.DataFrame(statuses))

    features = feature_columns(train)
    train_dice = train[features + [TARGET]].copy()
    continuous = [c for c in features if train[c].dtype != "object"]
    data_interface = dice_ml.Data(dataframe=train_dice, continuous_features=continuous, outcome_name=TARGET)
    adapter = ThresholdModelAdapter(model, threshold)
    model_interface = dice_ml.Model(model=adapter, backend="sklearn", model_type="classifier")
    explainer = dice_ml.Dice(data_interface, model_interface, method="random")
    vary = "all" if level == 1 else [
        c for group in ("perception_experience", "job_structure_context", "directly_actionable")
        for c in FEATURE_GROUPS[group]
    ]
    permitted = _dice_permitted_ranges(train[features])
    frames, statuses = [], []
    for _, employee in cohort.iterrows():
        last_error = ""
        for attempt in range(retry_count + 1):
            try:
                np.random.seed(seed + attempt)
                # DiCE creates a tqdm bar for every single-row request even with
                # verbose=False. Capture it so a cohort run remains readable.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    generated = explainer.generate_counterfactuals(
                        employee[features].to_frame().T, total_CFs=total_cfs, desired_class=0,
                        features_to_vary=vary, permitted_range=permitted, random_seed=seed + attempt,
                        posthoc_sparsity_param=0,
                    )
                cf = generated.cf_examples_list[0].final_cfs_df
                if cf is None:
                    cf = pd.DataFrame(columns=features)
                cf = cf.drop(columns=[TARGET], errors="ignore").drop_duplicates()
                before = float(model.predict_proba(employee[features].to_frame().T)[:, 1][0])
                if not cf.empty:
                    scores = model.predict_proba(cf[features])[:, 1]
                    rows = []
                    for sequence, ((_, candidate), score) in enumerate(zip(cf.iterrows(), scores), 1):
                        changed = _changed_features(employee, candidate, features)
                        violations = _diagnostic_constraint_violations(
                            employee, candidate, train, salary_cap_percentile, minimum_group_size
                        )
                        rows.append({
                            "employee_id": employee[ID_COLUMN], "constraint_level": level,
                            "scenario_id": f"{employee[ID_COLUMN]}-L{level}-{sequence:03d}",
                            "score_before": before, "score_after": float(score),
                            "changed_features": "|".join(changed),
                            "changes_json": _changes_json(employee, candidate, changed),
                            "n_changes": len(changed),
                            "salary_increase_pct": np.nan, "stock_option_increment": np.nan,
                            "overtime_change": "diagnostic", "model_valid": bool(score < threshold),
                            "hr_feasible": False, "violations": "|".join(violations),
                        })
                    frames.append(pd.DataFrame(rows, columns=SCENARIO_COLUMNS))
                statuses.append({"employee_id": employee[ID_COLUMN], "constraint_level": level, "success": True, "error": ""})
                break
            except Exception as exc:  # retry once, then preserve failure explicitly
                last_error = str(exc)
        else:
            no_counterfactual = is_no_counterfactual_result(last_error)
            statuses.append({
                "employee_id": employee[ID_COLUMN], "constraint_level": level,
                "success": no_counterfactual,
                "error": "" if no_counterfactual else last_error,
            })
    nonempty = [frame for frame in frames if not frame.empty]
    scenarios = pd.concat(nonempty, ignore_index=True) if nonempty else empty_scenarios()
    return CounterfactualRun(scenarios, pd.DataFrame(statuses))
