"""CSV and figure helpers kept separate from analytical decisions."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from .data_modeling import FEATURE_GROUPS, TARGET


def ensure_output_dirs(base: str | Path) -> dict[str, Path]:
    base = Path(base)
    paths = {name: base / name for name in ("tables", "figures", "scenarios", "models")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def feature_grouping_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature_group": group, "feature": feature}
        for group, features in FEATURE_GROUPS.items() for feature in features
    ])


def create_eda_outputs(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    sns.set_theme(style="whitegrid")
    overtime = df.groupby("OverTime")[TARGET].agg(["mean", "count"]).reset_index().rename(columns={"mean": "attrition_rate"})
    roles = df.groupby("JobRole")[TARGET].agg(["mean", "count"]).reset_index().rename(columns={"mean": "attrition_rate"})
    write_csv(overtime, tables / "attrition_by_overtime.csv")
    write_csv(roles, tables / "attrition_by_job_role.csv")

    plots = [
        ("target_distribution.png", lambda ax: sns.countplot(data=df, x=TARGET, ax=ax), "Attrition distribution"),
        ("monthly_income_distribution.png", lambda ax: sns.histplot(data=df, x="MonthlyIncome", bins=30, ax=ax), "Monthly income distribution"),
        ("attrition_by_overtime.png", lambda ax: sns.barplot(data=overtime, x="OverTime", y="attrition_rate", ax=ax), "Attrition rate by overtime"),
        ("attrition_by_job_role.png", lambda ax: sns.barplot(data=roles, y="JobRole", x="attrition_rate", ax=ax), "Attrition rate by job role"),
    ]
    for filename, draw, title in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)


def plot_model_outputs(y_true: pd.Series, scores: np.ndarray, threshold: float,
                       cm: np.ndarray, figures: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision–recall curve")
    fig.tight_layout(); fig.savefig(figures / "precision_recall_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png", dpi=160); plt.close(fig)


def plot_importance(feature_importance: pd.DataFrame, group_importance: pd.DataFrame, figures: Path) -> None:
    top = feature_importance.head(15).sort_values("mean_abs_shap")
    for frame, label, filename in [
        (top, "original_feature", "shap_feature_importance.png"),
        (group_importance.sort_values("mean_abs_shap"), "feature_group", "shap_group_importance.png"),
    ]:
        display_frame = frame.copy()
        if label == "feature_group":
            display_frame[label] = display_frame[label].map({
                "job_structure_context": "Job structure / context",
                "historical": "Historical",
                "directly_actionable": "Directly actionable",
                "perception_experience": "Perception / experience",
                "immutable": "Immutable",
            }).fillna(display_frame[label])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(display_frame[label], display_frame["mean_abs_shap"])
        title = "SHAP Feature Importance" if label == "original_feature" else "SHAP Group Importance"
        ax.set(xlabel="Mean |SHAP value|", title=title)
        fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)


def plot_assessment(assessment: pd.DataFrame, figures: Path) -> None:
    completed = assessment[assessment["analysis_status"] == "completed"]
    if completed.empty:
        return
    counts = completed["final_group"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax)
    ax.set(xlabel="Final group", ylabel="Employees", title="HR intervention assessment groups")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figures / "final_group_distribution.png", dpi=160); plt.close(fig)


def plot_supporting_analyses(
    found_rate: pd.DataFrame,
    violations: pd.DataFrame,
    levers: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    cap_sensitivity: pd.DataFrame,
    overtime_sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    figures: Path,
) -> None:
    """Create publication-ready plots for the proposal's supporting analyses."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.barplot(data=found_rate, x="constraint_level", y="found_rate", color="#4472C4", ax=ax)
    ax.set(xlabel="Constraint level", ylabel="Employees with model-valid scenario",
           title="Counterfactual found rate by constraint level", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout(); fig.savefig(figures / "counterfactual_found_rate.png", dpi=180); plt.close(fig)

    top_violations = violations.head(8).sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top_violations["violation"], top_violations["count"], color="#C55A11")
    ax.set(xlabel="Violation occurrences", title="Reasons model-valid diagnostic scenarios were rejected")
    fig.tight_layout(); fig.savefig(figures / "rejection_reasons.png", dpi=180); plt.close(fig)

    lever_plot = levers.copy()
    lever_plot["hr_lever"] = lever_plot["hr_lever"].map({
        "MonthlyIncome": "Monthly income",
        "StockOptionLevel": "Stock-option level",
        "OverTime": "Overtime",
    }).fillna(lever_plot["hr_lever"])
    lever_plot["scope"] = lever_plot["scope"].map({
        "all feasible Level 3": "All feasible Level 3",
        "selected best": "Selected best",
    }).fillna(lever_plot["scope"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=lever_plot, x="hr_lever", y="share_of_scenarios", hue="scope", ax=ax)
    ax.set(xlabel="HR lever", ylabel="Share of scenarios", title="HR lever contribution")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Scope")
    fig.tight_layout(); fig.savefig(figures / "hr_lever_contribution.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=threshold_sensitivity, x="value", y="feasibility_rate", marker="o", ax=ax)
    ax.set(xlabel="High-risk threshold", ylabel="Level 3 feasibility rate",
           title="Sensitivity to the high-risk threshold", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout(); fig.savefig(figures / "sensitivity_threshold.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.lineplot(data=cap_sensitivity, x="value", y="feasibility_rate", marker="o", ax=ax)
    ax.set(xlabel="Salary-cap percentile", ylabel="Level 3 feasibility rate",
           title="Sensitivity to the salary cap", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout(); fig.savefig(figures / "sensitivity_salary_cap.png", dpi=180); plt.close(fig)

    overtime_plot = overtime_sensitivity.copy()
    overtime_plot["value"] = overtime_plot["value"].astype(str).str.capitalize()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.barplot(data=overtime_plot, x="value", y="feasibility_rate", color="#70AD47", ax=ax)
    ax.set(xlabel="Overtime policy", ylabel="Level 3 feasibility rate",
           title="Sensitivity to overtime availability", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout(); fig.savefig(figures / "sensitivity_overtime.png", dpi=180); plt.close(fig)

    completed_stability = stability[stability["status"] == "completed"]
    stability_plot = completed_stability.sort_values("seed").copy()
    stability_plot["seed"] = stability_plot["seed"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=stability_plot, x="seed", y="feasibility_rate", color="#4C72B0", ax=ax)
    ax.set(xlabel="Split seed", ylabel="Feasibility rate",
           title="Feasibility across stratified splits", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout(); fig.savefig(figures / "stability_by_seed.png", dpi=180); plt.close(fig)
