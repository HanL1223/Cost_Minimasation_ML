<<<<<<< HEAD
"""
Script to oversample y label to ensure we can have a balance depend variable
"""
import os
import pandas as pd
from abc import ABC, abstractmethod
from src.logger import get_logger  # import logger
from imblearn.over_sampling import SMOTE
import numpy as np

=======
import pandas as pd
from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from src.logger import get_logger
>>>>>>> d4dd01551e995811ce3f6003407eec72c4d584cd

logger = get_logger(__name__)


<<<<<<< HEAD
class OverSampler(ABC):
    """
    Abstract base class for oversampling strategies.
    """
    @abstractmethod
    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method to resample features X and target y.

        Returns:
        X_res, y_res: resampled features and target
        """
        pass


class SMOTEOverSampler(OverSampler):
    def __init__(self,random_state = 42,sampling_strategy = 1.0,k_neighbors =5):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Applying SMOTE oversampling with sampling_strategy={self.sampling_strategy}")
        smote = SMOTE(
            random_state=self.random_state, 
            sampling_strategy=self.sampling_strategy, 
            k_neighbors=self.k_neighbors
        )
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"Oversampling completed: {len(y_res) - len(y)} samples added")
        return X_res, y_res
    
class OverSamplerHandler:
    """
    2
    """
    def __init__(self, strategy_class: type[OverSampler], target_col: str, **strategy_kwargs):
        self._strategy = strategy_class(**strategy_kwargs)
        self.target_col = target_col

    def set_strategy(self, strategy_class: type[OverSampler], **strategy_kwargs):
        logger.info("Switching oversampling strategy.")
        self._strategy = strategy_class(**strategy_kwargs)

    def resample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a DataFrame and return a resampled DataFrame."""
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Only numeric columns for SMOTE
        num_cols = X.select_dtypes(include="number").columns
        X_num = X[num_cols].copy()

        # Replace inf/-inf with NaN
        X_num.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill missing values with median
        X_num.fillna(X_num.median(), inplace=True)

        # Resample
        X_res, y_res = self._strategy.resample(X_num, y)

        # Recombine into DataFrame
        df_resampled = pd.concat([X_res, y_res], axis=1)
        return df_resampled
=======
# -------------------------------
# Base Abstract Class
# -------------------------------
class DataSampler(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Abstract method to resample dataset."""
        pass


# -------------------------------
# SMOTE Oversampling
# -------------------------------
class SMOTESampler(DataSampler):
    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying SMOTE oversampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        smote = SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"SMOTE complete. Output size: {df_resampled.shape}")
        return df_resampled


# -------------------------------
# Random Undersampling
# -------------------------------
class UnderSampler(DataSampler):
    def __init__(self, random_state: int = 42, sampling_strategy: str = "auto"):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying Random Undersampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        rus = RandomUnderSampler(random_state=self.random_state,
                                 sampling_strategy=self.sampling_strategy)
        X_res, y_res = rus.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"Undersampling complete. Output size: {df_resampled.shape}")
        return df_resampled


# -------------------------------
# Hybrid (SMOTE + ENN)
# -------------------------------
class SMOTEENNSampler(DataSampler):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying SMOTEENN hybrid resampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        smoteenn = SMOTEENN(random_state=self.random_state)
        X_res, y_res = smoteenn.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"SMOTEENN complete. Output size: {df_resampled.shape}")
        return df_resampled

class SamplerFactory:
    @staticmethod
    def create(sampler_type: str, **kwargs) -> DataSampler:
        sampler_type = sampler_type.lower()
        if sampler_type == "smote":
            return SMOTESampler(**kwargs)
        elif sampler_type == "undersample":
            return UnderSampler(**kwargs)
        elif sampler_type == "smoteenn":
            return SMOTEENNSampler(**kwargs)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")


if __name__ == "__main__":
    pass
>>>>>>> d4dd01551e995811ce3f6003407eec72c4d584cd
