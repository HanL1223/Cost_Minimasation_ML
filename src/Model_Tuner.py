import yaml
import os
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from joblib import dump
from abc import ABC,abstractmethod
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Any, Dict
import optuna   

from src.Get_Logging_Config import get_logger  # import logger
from src.Model_Selector import CrossValidationEvaluation


logger = get_logger(__name__)


# #Scorer
# def minimum_vs_model_cost(y_true, y_pred, cost_params={'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}):
#     cm = confusion_matrix(y_true, y_pred)
#     TP = cm[1, 1]
#     FP = cm[0, 1]
#     FN = cm[1, 0]

#     min_cost = (TP + FN) * cost_params['fn_cost']
#     model_cost = (TP * cost_params['tp_cost'] + FP * cost_params['fp_cost'] + FN * cost_params['fn_cost'])
#     return min_cost / model_cost

# custom_scorer = make_scorer(minimum_vs_model_cost, greater_is_better=True)


# ------------------------------------------------------
class OptunaTuning:
    """
    Advanced hyperparameter tuning using Optuna optimization framework.

    This class integrates Optuna's Bayesian optimization for hyperparameter search.
    It supports YAML-based configuration, cost-sensitive scoring, and model persistence.
    """

    def __init__(self, config_dir: str = "config", n_trials: int = 50, cv_folds: int = 3,
                 random_state: int = 42):
        self.config_dir = config_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

    # --------------------------------------------------
    # Load Hyperparameter Search Space from YAML
    # --------------------------------------------------
    def load_param_space(self, model_name: str) -> Dict[str, Any]:
        config_path = os.path.join(self.config_dir, f"{model_name}.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file for {model_name} not found at {config_path}")

        with open(config_path, 'r',encoding='utf-8') as f:
            param_config = yaml.safe_load(f)

        logger.info(f"Loaded tuning configuration for {model_name} from {config_path}")
        return param_config

    # --------------------------------------------------
    # Objective Function for Optuna
    # --------------------------------------------------
    def _objective(self, trial, model_class, X, y, param_space: Dict[str, Any]):
        """Define Optuna's objective function."""

        # Dynamically sample parameters from YAML search space
        params = {}
        for key, value in param_space.items():
            if isinstance(value, dict):
                method = value.get('method', 'uniform')
                if method == 'uniform':
                    params[key] = trial.suggest_float(key, value['low'], value['high'])
                elif method == 'loguniform':
                    params[key] = trial.suggest_float(key, value['low'], value['high'], log=True)
                elif method == 'int':
                    params[key] = trial.suggest_int(key, value['low'], value['high'])
                elif method == 'categorical':
                    params[key] = trial.suggest_categorical(key, value['choices'])
                elif method == "float":
            # Use log-scale for learning_rate or other parameters if specified
                    log_scale = value.get("log", False)
                    params[key] = trial.suggest_float(key, value['low'], value['high'], log=log_scale)
                else:
                    raise ValueError(f"Unknown sampling method '{method}' in config for {key}.")
            else:
                params[key] = value  # static value (no tuning)

        model = model_class(**params)

        # Stratified CV
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Evaluate with cross-validation using custom scorer
        scorer = CrossValidationEvaluation._create_default_scorer()
        scores = cross_val_score(model, X, y, cv=kfold, scoring=scorer, n_jobs=-1)
        mean_score = np.mean(scores)
        logger.debug(f"Trial params: {params} | Mean Score: {mean_score:.4f}")

        return mean_score

    # --------------------------------------------------
    # Run Optimization
    # --------------------------------------------------
    def tune(self, model_name: str, model_class, X, y):
        """Run Optuna hyperparameter optimization."""
        param_space = self.load_param_space(model_name)

        logger.info(f"Starting Optuna optimization for {model_name} ({self.n_trials} trials)...")

        study = optuna.create_study(direction="maximize", study_name=f"{model_name}_optuna_tuning")

        study.optimize(
            lambda trial: self._objective(trial, model_class, X, y, param_space),
            n_trials=self.n_trials,
            n_jobs=1,
            show_progress_bar=True
        )

        logger.info(f"Best Score: {study.best_value:.4f}")
        logger.info(f"Best Params: {study.best_params}")

        best_model = model_class(**study.best_params)
        best_model.fit(X, y)

        return best_model, study.best_params, study.best_value

    # --------------------------------------------------
    # Save Tuned Model
    # --------------------------------------------------
    @staticmethod
    def save_tuned_model(name: str, model, path="../models/tuned"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{name}_optuna_tuned.pkl")
        dump(model, model_path)
        logger.info(f"Tuned model '{name}' saved to: {model_path}")
        return model_path


# ------------------------------------------------------
# Example Usage (if run directly)
# ------------------------------------------------------
if __name__ == "__main__":
    from xgboost import XGBClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Example dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create tuner
    tuner = OptunaTuning(config_dir="config", n_trials=30, cv_folds=3)

    # Tune model (assuming config/xgbclassifier.yaml exists)
    best_model, best_params, best_score = tuner.tune("xgbclassifier", XGBClassifier, X_train, y_train)

    # Save tuned model
    tuner.save_tuned_model("xgbclassifier", best_model)