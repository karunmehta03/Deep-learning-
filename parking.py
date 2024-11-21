import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# Load the trained model
pipeline = joblib.load(r"C:\Users\manan\Downloads\yelp_dataset\parking.joblib")

# Streamlit app
st.title("Business Open Prediction")

# Input fields
stars = st.number_input("Stars", min_value=0.0, max_value=5.0, step=0.1)
review_count = st.number_input("Review Count", min_value=0, step=1)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.0001)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.0001)
categories = st.text_input("Categories (comma separated)", "Doctors, Traditional Chinese Medicine")
hours = st.text_input("Hours", "None")

# Make prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'stars': [stars],
        'review_count': [review_count],
        'latitude': [latitude],
        'longitude': [longitude],
        'categories': [categories],
        'hours': [hours]
    })

    # Make prediction
    prediction = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)

    # Display results
    st.write(f"Prediction: {'Open' if prediction[0] == 1 else 'Closed'}")
    st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")
