from src.Data_Samplier import SamplerFactory
import pandas as pd
from src.Get_Logging_Config import get_logger


logger = get_logger(__name__)


def data_sampling_step(method,df,target_col)->pd.DataFrame:
    sample_type = method.lower()
    if sample_type in ["smote","undersample","smoteenn"]:
            scaler = SamplerFactory.create(sample_type)
            df_scaled = scaler.impute(df,target_col=target_col)
            logger.info(f"Applying sampling. using {sample_type}")
            return df_scaled
    else:
            raise ValueError(f"Unknown sampler type: {sample_type}")