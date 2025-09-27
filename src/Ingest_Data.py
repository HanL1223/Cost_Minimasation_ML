import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
import requests

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,source_path:str) -> pd.DataFrame:
        """Abstract method to ingest data"""
        pass

class CSVIngestor(DataIngestor):
     """Extract a .csv file from path as pandas DataFrame"""
     def ingest(self,source_path:str) -> pd.DataFrame:
         if not os.path.exists(source_path):
             raise FileNotFoundError(f"CSV file not found at {source_path}")
         return pd.read_csv(source_path)
     

class ZIPIngestor(DataIngestor):
    def ingest(self, source_path: str) -> pd.DataFrame:  # Changed parameter name
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"ZIP file not found at {source_path}")
            
        # Extract to same directory as ZIP file
        extract_dir = os.path.dirname(source_path)
        with zipfile.ZipFile(source_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the extracted CSV
        extracted_files = os.listdir(extract_dir)
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted files")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in the extracted files")
        
        return pd.read_csv(os.path.join(extract_dir, csv_files[0]))


class APIIngestor(DataIngestor):
    """Extract Data via API"""
    def ingest(self,source_path:str) -> pd.DataFrame:
        print(f"Fetching data from API endpoint Point{source_path}")
        try:
            response = requests.get(source_path)
            response.raise_for_status()
            data = response.json()

            return pd.DataFrame(data)
        except requests.exceptions.RequestException as e:
            # Handle network errors, invalid URLs, etc.
            raise ConnectionError(f"Failed to fetch data from API: {source_path}") from e
        

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(source_path: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on the source path.
        It can handle URLs (http/https) and local file paths (.zip, .csv).
        """
        # Check if the source is a URL
        if source_path.lower().startswith(('http://', 'https://')):
            return APIIngestor()
        
        # Otherwise, treat it as a file path and check the extension
        file_extension = os.path.splitext(source_path)[1]
        
        if file_extension == ".zip":
            return ZIPIngestor()
        elif file_extension == ".csv":
            return CSVIngestor()
        else:
            raise ValueError(f"No ingestor available for source: {source_path}")
        
if __name__ == "__main__":
    pass