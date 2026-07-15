"""SHAP attribution on the selected leakage-safe sklearn pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_modeling import FEATURE_GROUPS, feature_columns


def _original_feature(transformed_name: str, original_features: list[str]) -> str:
    name = transformed_name.split("__", 1)[-1]
    exact = [f for f in original_features if name == f]
    if exact:
        return exact[0]
    matches = [f for f in original_features if name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


def compute_shap_importance(model: Any, train: pd.DataFrame, test: pd.DataFrame,
                            seed: int = 42, max_background: int = 150) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required; install requirements.txt") from exc

    features = feature_columns(train)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    background_raw = train[features].sample(min(max_background, len(train)), random_state=seed)
    background = preprocessor.transform(background_raw)
    values = preprocessor.transform(test[features])
    if hasattr(background, "toarray"):
        background = background.toarray()
    if hasattr(values, "toarray"):
        values = values.toarray()
    names = preprocessor.get_feature_names_out().tolist()

    model_name = estimator.__class__.__name__
    if "Forest" in model_name or "XGB" in model_name:
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(values)
    elif "LogisticRegression" in model_name:
        explainer = shap.LinearExplainer(estimator, background)
        explanation = explainer(values)
    else:
        explainer = shap.Explainer(estimator.predict_proba, background, feature_names=names)
        explanation = explainer(values)

    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if shap_values.shape[1] != len(names):
        raise RuntimeError("Unexpected SHAP output dimensions")
    original = [_original_feature(name, features) for name in names]
    transformed = pd.DataFrame({
        "transformed_feature": names,
        "original_feature": original,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    })
    feature_importance = transformed.groupby("original_feature", as_index=False)["mean_abs_shap"].sum()
    feature_importance = feature_importance.sort_values("mean_abs_shap", ascending=False)
    group_lookup = {feature: group for group, fields in FEATURE_GROUPS.items() for feature in fields}
    feature_importance["feature_group"] = feature_importance["original_feature"].map(group_lookup).fillna("unassigned")
    group_importance = feature_importance.groupby("feature_group", as_index=False)["mean_abs_shap"].sum()
    group_importance = group_importance.sort_values("mean_abs_shap", ascending=False)

    # Aggregate one-hot components back to original features for interpretable
    # employee-level case studies. Signed SHAP values are summed because they
    # describe the net direction of the original feature's contribution.
    local_transformed = pd.DataFrame(shap_values, columns=names, index=test.index)
    local = pd.DataFrame(index=test.index)
    for feature in features:
        columns = [name for name, source in zip(names, original) if source == feature]
        local[feature] = local_transformed[columns].sum(axis=1)
    local.insert(0, "employee_id", test["EmployeeNumber"].to_numpy())
    return feature_importance, group_importance, local
