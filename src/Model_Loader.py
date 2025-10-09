import os
from joblib import load
import pandas as pd
from src.Get_Logging_Config import get_logger  # Use your existing logger

logger = get_logger(__name__)

class ModelLoader:
    """
    Utility class to load a saved ML model from a .pkl file.
    """

    @staticmethod
    def load_model(model_path: str):
        """
        Load a model from a .pkl file.

        Parameters
        ----------
        model_path : str
            Full path to the saved model file.

        Returns
        -------
        model : object
            Loaded scikit-learn/XGBoost model.
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model

    @staticmethod
    def predict(model, X: pd.DataFrame):
        """
        Make predictions using the loaded model.

        Parameters
        ----------
        model : object
            Trained ML model.
        X : pd.DataFrame
            Feature dataframe for prediction.

        Returns
        -------
        predictions : np.ndarray
            Predicted labels or probabilities (depending on the model).
        """
        try:
            predictions = model.predict(X)
            logger.info(f"Predictions made on input of shape {X.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e


# Example usage
if __name__ == "__main__":
    model_path = r"models/tuned/xgbclassifier_tuned.pkl"
    loader = ModelLoader()
    model = loader.load_model(model_path)
    
    # Example prediction
    # X_new = pd.DataFrame(...)  # your new data here
    # y_pred = loader.predict(model, X_new)
