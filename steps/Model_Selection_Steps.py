# steps/model_selection_step.py
from zenml import step
from typing import Any, Tuple
from src.Model_Selector import ModelEvaluator, CrossValidationEvaluation  # adjust import
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

@step
def model_selection_step(X_train, y_train) -> Tuple[str, Any]:
    """
    Evaluates multiple models and returns the best one.
    """
    models = [
        ("log", LogisticRegression(solver="newton-cg", random_state=42)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
        ("XGBClassifier", XGBClassifier(random_state=42, eval_metric="logloss", device='cpu'))
    ]

    strategy = CrossValidationEvaluation(n_splits=5, random_state=42)
    evaluator = ModelEvaluator(strategy=strategy)
    results = evaluator.evaluate_models(models=models, X=X_train, y=y_train)

    best_model_name, best_model = evaluator.get_best_model(models)
    return best_model_name, best_model,results
