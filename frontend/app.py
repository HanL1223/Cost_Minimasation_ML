# app.py
import streamlit as st
import pandas as pd
from src.Model_Loader import ModelLoader
import os

st.title("Cost Prediction App")

# Upload data: CSV or JSON
uploaded_file = st.file_uploader(
    "Upload your data (CSV or JSON)", type=["csv", "json"]
)

if uploaded_file is not None:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        input_data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        input_data = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    st.write("Input Data Preview:")
    st.dataframe(input_data.head())

    # Load trained model
    model_path = os.path.join("models", "tuned", "XGBClassifier_optuna_tuned.pkl")
    model = ModelLoader.load_model(model_path)

    # Perform inference
    st.write("Running inference...")
    predictions = ModelLoader.predict(model, input_data)
    st.write("Predictions:")
    st.dataframe(predictions)
