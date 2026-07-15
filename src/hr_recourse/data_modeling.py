"""Data validation, leakage-safe splitting, model selection, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Attrition"
ID_COLUMN = "EmployeeNumber"
EXCLUDED_FEATURES = ["EmployeeCount", "StandardHours", "Over18", ID_COLUMN]

FEATURE_GROUPS: dict[str, list[str]] = {
    "immutable": ["Age", "Gender", "MaritalStatus"],
    "historical": [
        "Education", "EducationField", "NumCompaniesWorked", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager",
    ],
    "perception_experience": [
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ],
    "job_structure_context": [
        "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
        "HourlyRate", "JobLevel", "JobRole", "MonthlyRate",
        "PercentSalaryHike", "PerformanceRating",
    ],
    "directly_actionable": ["MonthlyIncome", "StockOptionLevel", "OverTime"],
}


@dataclass
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    validation_scores: np.ndarray
    validation_ap: float


def load_and_validate_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IBM data, validate research-critical fields, and return quality checks."""
    df = pd.read_csv(path)
    required = {TARGET, ID_COLUMN, "MonthlyIncome", "StockOptionLevel", "OverTime", "JobRole", "JobLevel"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not set(df[TARGET].dropna().unique()).issubset({"Yes", "No", 0, 1}):
        raise ValueError("Attrition must contain only Yes/No or 0/1")

    checks = [
        ("row_count", len(df)),
        ("column_count", df.shape[1]),
        ("missing_cells", int(df.isna().sum().sum())),
        ("duplicate_rows", int(df.duplicated().sum())),
        ("duplicate_employee_ids", int(df[ID_COLUMN].duplicated().sum())),
        ("invalid_years_current_role", int((df["YearsInCurrentRole"] > df["YearsAtCompany"]).sum())),
        ("invalid_years_with_manager", int((df["YearsWithCurrManager"] > df["YearsAtCompany"]).sum())),
        ("invalid_years_since_promotion", int((df["YearsSinceLastPromotion"] > df["YearsAtCompany"]).sum())),
        ("nonpositive_monthly_income", int((df["MonthlyIncome"] <= 0).sum())),
    ]
    quality = pd.DataFrame(checks, columns=["check", "value"])
    df = df.copy()
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).fillna(df[TARGET]).astype(int)
    return df, quality


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET and c not in EXCLUDED_FEATURES]


def split_data(
    df: pd.DataFrame, seed: int = 42, train_size: float = 0.60,
    validation_size: float = 0.20, test_size: float = 0.20,
) -> DataSplits:
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Split proportions must sum to 1")
    train, remainder = train_test_split(
        df, train_size=train_size, stratify=df[TARGET], random_state=seed,
    )
    validation, test = train_test_split(
        remainder,
        train_size=validation_size / (validation_size + test_size),
        stratify=remainder[TARGET], random_state=seed,
    )
    return DataSplits(
        train=train.sort_index().copy(), validation=validation.sort_index().copy(), test=test.sort_index().copy()
    )


def split_summary(splits: DataSplits) -> pd.DataFrame:
    rows = []
    for name, frame in vars(splits).items():
        rows.append({"split": name, "n_rows": len(frame), "attrition_count": int(frame[TARGET].sum()),
                     "attrition_rate": float(frame[TARGET].mean())})
    return pd.DataFrame(rows)


def build_preprocessor(train: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    features = feature_columns(train)
    categorical = [c for c in features if train[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
        ("numeric", Pipeline(numeric_steps), numeric),
    ])


def candidate_models(train: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    y = train[TARGET]
    negative, positive = np.bincount(y)
    scale_pos_weight = negative / max(positive, 1)
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            # DiCE performs many tiny predictions; n_jobs=1 avoids process/thread
            # scheduling overhead that dominates those calls on some platforms.
            n_jobs=1, random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", scale_pos_weight=scale_pos_weight,
            n_jobs=1, random_state=seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not be loaded. Install requirements.txt and, on macOS, "
            "install the OpenMP runtime (for example: brew install libomp)."
        ) from exc
    return {
        name: Pipeline([
            ("preprocessor", build_preprocessor(train, name == "Logistic Regression")),
            ("model", model),
        ]) for name, model in models.items()
    }


def fit_and_compare_models(train: pd.DataFrame, validation: pd.DataFrame, seed: int = 42) -> tuple[ModelResult, pd.DataFrame]:
    features = feature_columns(train)
    results: list[ModelResult] = []
    rows = []
    for order, (name, pipeline) in enumerate(candidate_models(train, seed).items()):
        pipeline.fit(train[features], train[TARGET])
        scores = pipeline.predict_proba(validation[features])[:, 1]
        ap = average_precision_score(validation[TARGET], scores)
        validation_threshold, _ = choose_threshold(validation[TARGET], scores)
        validation_pred = scores >= validation_threshold
        results.append(ModelResult(name, pipeline, scores, float(ap)))
        rows.append({
            "model": name,
            "validation_average_precision": ap,
            "validation_threshold": validation_threshold,
            "validation_precision": precision_score(validation[TARGET], validation_pred, zero_division=0),
            "validation_recall": recall_score(validation[TARGET], validation_pred, zero_division=0),
            "validation_f1": f1_score(validation[TARGET], validation_pred, zero_division=0),
            "validation_f2": fbeta_score(validation[TARGET], validation_pred, beta=2, zero_division=0),
            "simplicity_order": order,
        })
    comparison = pd.DataFrame(rows)
    comparison["rounded_ap"] = comparison["validation_average_precision"].round(4)
    winner_name = comparison.sort_values(["rounded_ap", "simplicity_order"], ascending=[False, True]).iloc[0]["model"]
    comparison["selected"] = comparison["model"].eq(winner_name)
    winner = next(r for r in results if r.name == winner_name)
    return winner, comparison.drop(columns="rounded_ap")


def choose_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray, minimum: float = 0.05,
                     maximum: float = 0.95, step: float = 0.01) -> tuple[float, pd.DataFrame]:
    thresholds = np.round(np.arange(minimum, maximum + step / 2, step), 10)
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(["f2", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return float(best["threshold"]), table


def classification_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float,
                           split: str) -> tuple[pd.DataFrame, np.ndarray]:
    pred = scores >= threshold
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    row = {
        "split": split, "threshold": threshold,
        "average_precision": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return pd.DataFrame([row]), cm


def predict_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[feature_columns(frame)])[:, 1]
