import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
def load_model():
    with open('breast_cancer_new.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define the feature columns
feature_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                   'mean smoothness', 'mean compactness', 'mean concavity',
                   'mean concave points', 'mean symmetry', 'mean fractal dimension',
                   'radius error', 'texture error', 'perimeter error', 'area error',
                   'smoothness error', 'compactness error', 'concavity error',
                   'concave points error', 'symmetry error', 'fractal dimension error',
                   'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                   'worst smoothness', 'worst compactness', 'worst concavity',
                   'worst concave points', 'worst symmetry', 'worst fractal dimension']

# Initialize the model
model = load_model()

# Streamlit app
st.title("Breast Cancer Prediction")

# Create input fields for the features
input_data = []
for feature in feature_columns:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data], columns=feature_columns)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction
    if prediction[0] == 0:
        st.write("The model predicts: **Not Dangerous**")
    else:
        st.write("The model predicts: **Dangerous**")

    # Display the prediction probabilities
    st.write(f"Prediction probabilities: {prediction_proba}")
