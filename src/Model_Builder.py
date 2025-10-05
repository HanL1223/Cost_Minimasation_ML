import logging
from abc import ABC, abstractmethod
from typing import Any
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
from src.logger import get_logger  # import logger
import optuna   

logger = get_logger(__name__)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """
        Abstract method to build and train a model.
        """
        pass

# Concrete Strategy: Use a pre-tuned XGBClassifier
class XGBClassifierStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Loads a tuned XGBClassifier model and fits it on the provided training data.
        """
        model_path = "models/tuned/Xgboost_tuned.pkl"

        logging.info(f"Loading tuned XGBClassifier from {model_path}")
        model = joblib.load(model_path)

        if not isinstance(model, XGBClassifier):
            raise TypeError("Loaded model is not an instance of XGBClassifier.")

        logging.info("Training the loaded XGBClassifier on the provided training data.")
        model.fit(X_train, y_train)

        logging.info("Model training complete.")
        return model
