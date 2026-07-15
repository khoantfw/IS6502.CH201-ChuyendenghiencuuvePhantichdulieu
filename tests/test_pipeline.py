from unittest.mock import patch

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import hr_recourse.data_modeling as dm
import run_pipeline


def lightweight_candidates(train, seed):
    result = {}
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        result[name] = Pipeline([
            ("preprocessor", dm.build_preprocessor(train, True)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=seed)),
        ])
    return result


def test_end_to_end_smoke_creates_main_schemas(tmp_path):
    source, _ = dm.load_and_validate_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    fixture = source.groupby(dm.TARGET, group_keys=False).sample(n=70, random_state=7)
    fixture[dm.TARGET] = fixture[dm.TARGET].map({1: "Yes", 0: "No"})
    data_path = tmp_path / "fixture.csv"
    fixture.to_csv(data_path, index=False)
    output = tmp_path / "outputs"
    config = {
        "data_path": str(data_path), "output_dir": str(output), "primary_seed": 42,
        "stability_seeds": [], "train_size": .6, "validation_size": .2, "test_size": .2,
        "threshold": {"minimum": .05, "maximum": .95, "step": .05},
        "counterfactuals": {"dice_total": 1, "retry_count": 0,
            "salary_increments": [0, .1, .3], "salary_cap_percentile": 75,
            "minimum_salary_group_size": 10, "overtime_policy": "allowed"},
        "sensitivity": {"threshold_offsets": [0], "salary_cap_percentiles": [75], "overtime_policies": ["allowed"]},
        "run": {"run_shap": False, "run_dice": False, "run_stability": False,
                "run_sensitivity": False, "generate_figures": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with patch("hr_recourse.data_modeling.candidate_models", lightweight_candidates):
        run_pipeline.run(config_path)
    expected = {
        "final_model_metrics.csv", "hr_intervention_feasibility_assessment.csv",
        "model_comparison.csv", "split_summary.csv",
    }
    assert expected.issubset({p.name for p in (output / "tables").iterdir()})
    assessment = pd.read_csv(output / "tables" / "hr_intervention_feasibility_assessment.csv")
    assert set(["employee_id", "final_group", "analysis_status"]).issubset(assessment.columns)
    comparison = pd.read_csv(output / "tables" / "model_comparison.csv")
    assert {"validation_average_precision", "validation_precision", "validation_recall",
            "validation_f1", "validation_f2"}.issubset(comparison.columns)
    assert (output / "scenarios" / "level_3_grid.csv").exists()
