import o
import pandas as pd
from abc import ABC, abstractmethod
from src.logger import get_logger  # import logger

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
    def handle(self,df:pd.DataFrame) -> pd.DataFrame:
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
    
