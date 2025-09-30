from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic_settings import BaseSettings, SettingsConfigDict 
from typing import AsyncGenerator
import numpy as np


CURRENT_FILE_PATH = os.path.abspath(__file__) 

PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)) 


# --- Configuration ---
# Update paths to point to the 'data' folder in the root directory
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data","raw\\" "Train.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

app = FastAPI(title="Local Data Split API")

# Global DataFrame cache
DF_TRAIN = None
DF_TEST = None

def load_and_split_data():
    """
     Loads the CSV from the local file path defined in configuration, 
    performs the train/test split, and caches the results.
    """
    global DF_TRAIN , DF_TEST

    if DF_TRAIN is not None and DF_TRAIN is not None:
        print("Use existing cache")
        return
    print(f"Server loading and splitting data from: {LOCAL_DATA_PATH}")

    try:
        if not os.path.exists(LOCAL_DATA_PATH):
            raise FileNotFoundError(f"WARRING:csv file not found  at {LOCAL_DATA_PATH}")
        df_full = pd.read_csv(LOCAL_DATA_PATH)


        DF_TRAIN, DF_TEST = train_test_split(
            df_full,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"Data ready. Train set: {len(DF_TRAIN)} rows, Test set: {len(DF_TEST)} rows.")

    except Exception as e:
        print(f"WARRING: Could not load or split data.")
        print(e)
        raise RuntimeError("Data Ingestion Failure") from e
    
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # --- Startup Logic ---
    # Code before 'yield' runs on application startup
    print("Application startup sequence initiated.")
    load_and_split_data() # Call your startup function here
    print("Startup complete. Application is ready to receive requests.")

    yield # This is where the application starts running

    # --- Shutdown Logic (Optional) ---
    # Code after 'yield' runs on application shutdown
    print("Application shutdown sequence initiated.")
    # You can add cleanup code here, e.g., closing database connections
    print("Shutdown complete.")


app = FastAPI(lifespan=lifespan)



@app.get("/train", tags=["Data"], summary="Get Training Data")
async def get_train_data():
    """Returns the training dataset as JSON-safe numeric types."""
    
    if DF_TRAIN is None:
        return {"error": "Training data not loaded"}, 500

    df_safe = DF_TRAIN.copy()

    # Convert numeric columns to float
    numeric_cols = df_safe.select_dtypes(include=['number']).columns
    df_safe[numeric_cols] = df_safe[numeric_cols].astype(float)

    # Replace NaN and infinite values with 0 (or another value)
    df_safe[numeric_cols] = df_safe[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_safe[numeric_cols] = df_safe[numeric_cols].fillna(0.0)

    return df_safe.to_dict(orient='records')


@app.get("/test", tags=["Data"], summary="Get Test Data")
async def get_test_data():
    """Returns the test dataset as a list of JSON records (safely converted)."""
    if DF_TEST is None:
        return {"error": "Test data not loaded"}, 500
        
    # Convert all columns to strings
    df_safe = DF_TEST.copy()
    numeric_cols = df_safe.select_dtypes(include=['number']).columns
    df_safe[numeric_cols] = df_safe[numeric_cols].astype(float)
    
    return df_safe.to_dict(orient='records')

# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure you are running uvicorn from the project root for pathing to work reliably
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)