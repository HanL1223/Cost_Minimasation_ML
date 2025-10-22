# steps/data_split_step.py
from zenml import step
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

@step
def data_split_step(df: pd.DataFrame, target_col: str,test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into train/test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
