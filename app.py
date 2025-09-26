%%writefile app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lr_model = joblib.load("lr_model.pkl")

st.title("💖 Heart Disease 10-Year Risk Prediction")

# User inputs
male = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120, 30)
education = st.number_input("Education Level (1-5)", 1, 5, 3)
currentSmoker = st.selectbox("Current Smoker?", ["No", "Yes"])
cigsPerDay = st.number_input("Cigarettes per day", 0, 100, 0)
BPMeds = st.selectbox("On Blood Pressure Medication?", ["No", "Yes"])
prevalentStroke = st.selectbox("Previous Stroke?", ["No", "Yes"])
prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
totChol = st.number_input("Total Cholesterol", 100, 400, 200)
sysBP = st.number_input("Systolic BP", 80, 250, 120)
diaBP = st.number_input("Diastolic BP", 40, 150, 80)
BMI = st.number_input("BMI", 10.0, 50.0, 25.0)
heartRate = st.number_input("Heart Rate", 40, 200, 70)
glucose = st.number_input("Glucose", 50, 300, 100)

# Convert categorical to numeric
male_val = 1 if male=="Male" else 0
currentSmoker_val = 1 if currentSmoker=="Yes" else 0
BPMeds_val = 1 if BPMeds=="Yes" else 0
prevalentStroke_val = 1 if prevalentStroke=="Yes" else 0
prevalentHyp_val = 1 if prevalentHyp=="Yes" else 0
diabetes_val = 1 if diabetes=="Yes" else 0

# Build feature array
input_features = np.array([[male_val, age, education, currentSmoker_val, cigsPerDay,
                            BPMeds_val, prevalentStroke_val, prevalentHyp_val,
                            diabetes_val, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

# Model selection
model_choice = st.selectbox("Choose model", ["RandomForest", "XGBoost", "LogisticRegression"])
model_dict = {
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
    "LogisticRegression": lr_model
}

# Prediction
if st.button("Predict"):
    model = model_dict[model_choice]
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted 10-Year CHD Risk: {'High' if prediction==1 else 'Low'}")
