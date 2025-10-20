# steps/model_tuning_step.py
from zenml import step
from typing import Any, Tuple
import pandas as pd
from src.Model_Evaluator import ClassificationModelEvaluator 
from pathlib import Path
import json
from datetime import datetime



@step
def model_evaluation_steps(model,X_test,y_test,task = 'classification',model_path = '/Users/hanli/cost_ml_202509/Cost_Minimasation_ML/models/tuned/xgbclassifier_optuna_tuned.pkl'):
    eval_map = {
        "classification": ClassificationModelEvaluator()
       # "regression": 
    }
    if task not in eval_map:
        raise ValueError(f"Invalid strategy '{task}'. Choose from {list(eval_map.keys())}")
    #Currently for classification task only and should expand to regression and more
    evaluator = eval_map[task]
    model = evaluator.load_model(model_path)
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    metrics_df = pd.DataFrame([metrics])

    # --- Prepare output directory ---
    step_dir = Path(__file__).resolve()
    project_root = step_dir.parents[1]
    result_dir = project_root / "result" / "validation"
    result_dir.mkdir(parents=True, exist_ok=True)

    # --- Save with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = result_dir / f"result_{timestamp}.json"
    csv_path = result_dir / f"result_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    metrics_df.to_csv(csv_path, index=False)

    print(f"\Metrics saved to: {json_path} and {csv_path}")

    # Optional: print nicely
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")




