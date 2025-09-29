import streamlit as st
import pandas as pd
import requests

st.title("ML Model Prediction Interface")

# Option 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)
    
    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=df.to_dict(orient="records"))
        predictions = response.json()
        st.write("Predictions:")
        st.write(predictions)

# Option 2: Input manually
st.subheader("Or input manually")
num_features = 5  # adjust based on your model
input_data = {}
for i in range(1, num_features + 1):
    input_data[f"V{i}"] = st.number_input(f"V{i}")

if st.button("Predict Single Sample"):
    df_manual = pd.DataFrame([input_data])
    response = requests.post("http://127.0.0.1:8000/predict", json=df_manual.to_dict(orient="records"))
    prediction = response.json()
    st.write("Prediction:")
    st.write(prediction)