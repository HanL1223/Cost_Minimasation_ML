# steps/missing_value_step.py
from src.Missing_value_handling import MissingValueHandler, FillMissingValue, DropMissingValue
import pandas as pd

def missing_value_step(df: pd.DataFrame, method,strategy: str = "fill", ) -> pd.DataFrame:
    strategy_map = {
        "fill": FillMissingValue,
        "drop": DropMissingValue
    }

    if strategy not in strategy_map:
        raise ValueError(f"Invalid strategy '{strategy}'. Choose from {list(strategy_map.keys())}")

    handler = MissingValueHandler(strategy_map[strategy], method =method)
    return handler.handle_missing_values(df)
