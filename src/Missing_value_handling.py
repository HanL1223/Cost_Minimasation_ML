import os
import pandas as pd
from abc import ABC, abstractmethod
from src.Get_Logging_Config import get_logger  # import logger

logger = get_logger(__name__)

class MissingDataImputer(ABC):
    @abstractmethod
    def Impute(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for Data handling 
        """
        pass

class DropMissingValue(MissingDataImputer):
    """
    Initializes the DropMissingValues Strategy with specific parameters.

    Parameters:
    axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
    thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
    """

    def __init__(self,axis,thresh):
        self.axis= axis
        self.thresh = thresh
    def Impute(self,df:pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Dropping missing value  with axis= {self.axis} and thresh = {self.thresh}")
        df_clean = df.dropna(axis = self.axis,thresh=self.thresh)
        logger.info("Missing value dropped")
        return df_clean
    
class FillMissingValue(MissingDataImputer):
    def __init__(self,method = 'mean',fill_value = None):
        """
    Initializes the FillMissingValuesStrategy with a specific method or fill value.

    Parameters:
    method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
    fill_value (any): The constant value to fill missing values when method='constant'.
    """
        self.method = method
        self.fill_value = fill_value

    def Impute(self,df:pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Filling value with {self.method} strategy")
        df_cleaned =  df.copy()
        if self.method == 'mean':
            df_cleaned = df_cleaned.fillna(df_cleaned.select_dtypes('number').mean())
        elif self.method == 'median':
            df_cleaned = df_cleaned.fillna(df_cleaned.select_dtypes('number').median())
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        else:
            logger.warning(f"Unknown method '{self.method}'. No missing values handled.")
        logger.info("Missing values filled.")
        return df_cleaned


class MissingValueHandler:
    #This require  creating a missing dataimputer object first - we should just pass the class
    # def __init__ (self, strategy:MissingDataImputer):
    #     self._strategy = strategy

    def __init__ (self,strategy_class: type[MissingDataImputer],**strategy_kwargs):
        self._strategy = strategy_class(**strategy_kwargs)

    def _set_strategy(self, strategy_class: type[MissingDataImputer], **strategy_kwargs):
        """Internal helper for creating and setting the strategy."""
        self._strategy = strategy_class(**strategy_kwargs)

    def set_strategy(self, strategy_class: type[MissingDataImputer], **strategy_kwargs):
        """Public method to switch strategies dynamically."""
        logger.info("Switching missing value handling strategy.")
        self._set_strategy(strategy_class, **strategy_kwargs)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the missing value handling using the current strategy."""
        logger.info("Executing missing value handling strategy.")
        return self._strategy.Impute(df)
    

if __name__ == "__main__":
    pass 