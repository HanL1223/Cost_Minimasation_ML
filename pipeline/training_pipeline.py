from zenml import pipeline
from steps.Data_Ingestion_Steps import data_ingestion_step
from steps.Data_Split_Steps import data_split_step
from steps.Data_Sampling_Steps import data_sampling_step
from steps.Model_Selection_Steps import model_selection_step
from steps.Model_Tuning_Steps import model_tuning_step
from steps.Missing_data_Handling_Steps import missing_value_step
import os
import sys
import time
import traceback
from datetime import datetime
from src.Get_Logging_Config import get_logger
import joblib

logger = get_logger(__name__)

BASE_URL = "http://127.0.0.1:8000"
TRAIN_ENDPOINT = f"{BASE_URL}/train"
TEST_ENDPOINT = f"{BASE_URL}/test"

def main():
    start_time = time.time()
    logger.info("Start Training Pipeline")

    try:
        # === Define paths ===
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # === Step 1: Data Ingestion ===
        logger.info("📥 Starting data ingestion")
        raw_data = data_ingestion_step(TRAIN_ENDPOINT).ingest(TRAIN_ENDPOINT)
        logger.info(f"Data ingestion completed: {raw_data.shape}")

        # === Step 2: Missing Data Handling ===
        logger.info("🧩 Handling missing data...")
        clean_data = missing_value_step(raw_data,strategy='fill')
        logger.info(f"Missing data handled: {clean_data.shape}")

        # === Step 3: Sampling (if needed) ===
        logger.info("Performing data sampling...")
        sampler = data_sampling_step(method='smoteenn',df = clean_data,target_col='Target')
        sampled_data = sampler.sample(clean_data)
        logger.info(f"Sampling completed: {sampled_data.shape}")

        # === Step 4: Data Split ===
        logger.info(" Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = data_split_step(df = sampled_data,target_col = 'Target')
        logger.info(f"Split done: Train={len(X_train)}, Test={len(X_test)}")

        # === Step 5: Model Selection ===
        logger.info("🤖 Selecting the best model...")
        selector = ModelSelector()
        best_model_name, best_model, baseline_score = selector.select(X_train, y_train, X_test, y_test)
        logger.info(f"Best model: {best_model_name} with baseline score {baseline_score}")

        # === Step 6: Model Tuning ===
        logger.info("⚙️ Tuning best model...")
        tuner = ModelTuner(model=best_model)
        tuned_model, tuned_score = tuner.tune(X_train, y_train, X_test, y_test)
        logger.info(f"Model tuning done: {tuned_model.__class__.__name__} - Final score: {tuned_score}")

        # === Step 7: Save Artifacts ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(artifacts_dir, f"best_model_{timestamp}.pkl")
        joblib.dump(tuned_model, model_path)
        logger.info(f"💾 Model saved at {model_path}")

        # === Step 8: Summary ===
        elapsed = time.time() - start_time
        logger.info(f"🎉 Training pipeline completed in {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
