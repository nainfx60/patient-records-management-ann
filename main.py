import streamlit as st
import numpy as np
import pandas as pd
import os

# Title of the Application
st.title("🏥 Patient Records Management & Analytics Dashboard")
st.write("Enter the patient's record details below to forecast the Expected Patient Satisfaction Score (0-100).")

FILE_NAME = "patients.csv"

# Check if file exists before running the app logic
if not os.path.exists(FILE_NAME):
    st.error(f"❌ Dataset file '{FILE_NAME}' GitHub par nahi mili!")
else:
    # 1. Load data safely
    df = pd.read_csv(FILE_NAME)

    # Clean data: Parse dates to calculate Length of Stay automatically
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    df['length_of_stay'] = (df['departure_date'] - df['arrival_date']).dt.days
    
    # Extract Options for Categorical Dropdowns
    service_options = sorted(df['service'].unique().tolist())

    # 2. Pure Mathematical Modeling (Linear combination of ANN weights logic)
    df_encoded = df.copy()
    df_encoded['service_idx'] = df_encoded['service'].map({val: i for i, val in enumerate(service_options)})

    # Features (X): age, length_of_stay, service_idx
    X = df_encoded[['age', 'length_of_stay', 'service_idx']].values
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Target (y): satisfaction score
    y = df_encoded['satisfaction'].values

    # Calculate weights using Normal Equation
    try:
        beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_bias) @ y

    # 3. Creating User Input Interface
    st.subheader("Patient Admission Details")

    age_input = st.number_input("Patient Age", min_value=0, max_value=120, value=35)
    length_of_stay_input = st.number_input("Length of Stay (Total Days in Hospital)", min_value=1, max_value=60, value=5)
    service_input = st.selectbox("Department/Service Availed", service_options)

    # Prediction execution button
    if st.button("Predict Satisfaction Score"):
        # Map user input to numerical index
        service_mapped = service_options.index(service_input)
        
        # Structure the complete user vector with bias term (1)
        user_features = np.array([1, age_input, length_of_stay_input, service_mapped])
        
        # Dot product for mathematical prediction
        predicted_score = np.dot(user_features, beta)
        
        # Bound the score between 0 and 100
        final_score = max(0.0, min(100.0, float(predicted_score)))
        
        st.success(f"🔮 Predicted Patient Satisfaction Score: **{final_score:.1f} / 100**")
