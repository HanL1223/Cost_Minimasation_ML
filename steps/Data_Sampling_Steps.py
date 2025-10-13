from src.Data_Samplier import SamplerFactory
import pandas as pd


def data_sampling(method,df,target_col)->pd.DataFrame:
    sample_type = method.lower()
    if sample_type in ["smote","undersample","smoteenn"]:
            scaler = SamplerFactory.create(sample_type)
            df_scaled = scaler.impute(df,target_col=target_col)
            return df_scaled
    else:
            raise ValueError(f"Unknown sampler type: {sample_type}")