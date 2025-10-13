from zenml import step
import os
import pandas as pd
from src.Data_ingestor import DataIngestorFactory

@step
def data_ingestion_step(source:str) -> pd.DataFrame:
    """
    Ingest Data from given source and return a Pandas Dataframe
    get_data_ingestor: read source format to identify what ingestor to use, return ingestor object
    ingest: ingest data using the ingestor object
    """
    df = DataIngestorFactory.get_data_ingestor(source).ingest(source)
    return df