# steps/model_tuning_step.py
from zenml import step
from typing import Any, Tuple
from src.Model_Tuner import OptunaTuning  # adjust path if needed

@step
def model_tuning_step(
    best_model_name: str,
    best_model: Any,
    X_train,
    y_train
) -> Tuple[Any, dict, float]:
    """
    ZenML step to perform hyperparameter tuning using Optuna.
    Automatically uses the best model from model selection.
    """
    tuner = OptunaTuning(config_dir="config", n_trials=50, cv_folds=5)

    # Pass dynamic model name and class directly
    best_model_class = best_model.__class__

    tuned_model, best_params, best_score = tuner.tune(
        best_model_name.lower(),
        best_model_class,
        X_train,
        y_train
    )
    model_path = tuner.save_tuned_model(best_model_name,best_model)

    return tuned_model, best_params, best_score,model_path
