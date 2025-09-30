"""
Script to oversample y label to ensure we can have a balance depend variable
"""
import os
import pandas as pd
from abc import ABC, abstractmethod
from src.logger import get_logger  # import logger
from imblearn.over_sampling import SMOTE
import numpy as np


logger = get_logger(__name__)


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
