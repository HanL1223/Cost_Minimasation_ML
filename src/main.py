from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import uvicorn

# Determine the absolute path to the project root (where the 'data' folder is)
# This assumes the script is run from inside the 'src' folder or via a standard uvicorn command.
# os.getcwd() gets the current working directory, and we move up one level (..) to find 'data'

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

# ----------------------------------------------------------------------
# --- Data Loading and Splitting Logic (Stays the same logically) ---
# ----------------------------------------------------------------------
def load_and_split_data():
    """
    Loads the CSV from the local file path defined in configuration, 
    performs the train/test split, and caches the results.
    """
    global DF_TRAIN, DF_TEST
    
    if DF_TRAIN is not None and DF_TEST is not None:
        print("Using cached dataframes.")
        return

    print(f"Server loading and splitting data from: {LOCAL_DATA_PATH}")
    
    try:
        if not os.path.exists(LOCAL_DATA_PATH):
            raise FileNotFoundError(f"FATAL: CSV file not found at {LOCAL_DATA_PATH}. Check your path.")
            
        df_full = pd.read_csv(LOCAL_DATA_PATH)
        
        DF_TRAIN, DF_TEST = train_test_split(
            df_full,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"Data ready. Train set: {len(DF_TRAIN)} rows, Test set: {len(DF_TEST)} rows.")
        
    except Exception as e:
        print(f"FATAL DATA ERROR: Could not load or split data.")
        print(e)
        raise RuntimeError("Data Ingestion Failure") from e

# Event handler to execute the data loading logic when the server starts up
@app.on_event("startup")
async def startup_event():
    load_and_split_data()

# ------------------------------------
# --- API Endpoints (The Service) ---
# ------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"message": "Local Data API is running!", "train_shape": list(DF_TRAIN.shape), "test_shape": list(DF_TEST.shape)}


# --- Inside main.py, replace the current get_train_data and get_test_data functions ---

@app.get("/train", tags=["Data"], summary="Get Training Data")
async def get_train_data():
    """Returns the training dataset as a list of JSON records (safely converted)."""
    # 1. Ensure the DataFrame exists
    if DF_TRAIN is None:
        return {"error": "Training data not loaded"}, 500
    
    # 2. Convert all columns to strings to ensure JSON compatibility 
    # This prevents failures from numpy types, dates, etc.
    df_safe = DF_TRAIN.apply(lambda x: x.astype(str), axis=0)
    
    return df_safe.to_dict(orient='records')

@app.get("/test", tags=["Data"], summary="Get Test Data")
async def get_test_data():
    """Returns the test dataset as a list of JSON records (safely converted)."""
    if DF_TEST is None:
        return {"error": "Test data not loaded"}, 500
        
    # Convert all columns to strings
    df_safe = DF_TEST.apply(lambda x: x.astype(str), axis=0)
    
    return df_safe.to_dict(orient='records')

# -----------------------------------------------------------------------------------

# Optional: Allows running the file directly via `python src/main.py`
if __name__ == "__main__":
    # Ensure you are running uvicorn from the project root for pathing to work reliably
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)