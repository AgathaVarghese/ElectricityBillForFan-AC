import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set Page Config
st.set_page_config(page_title="AC Electric Bill Predictor", layout="centered")

st.title("⚡ AC Electric Bill Predictor & Visualizer")
st.write("Predict electricity bills and compare actual vs. predicted values from your dataset.")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

pipeline = load_model()

# ==========================================
# 1. DATASET UPLOAD & COMPARISON PLOT
# ==========================================
st.subheader("📊 Dataset & Comparison Plot")
uploaded_file = st.file_uploader("Upload 'Day5-Christ-College-ElectricBill_AC.xlsx' or CSV file:", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Data Preview", df.head())

    # Ensure required columns exist
    if 'AC_Units' in df.columns and 'Electric_Bill (INR)' in df.columns:
        # Generate Predictions for the dataset
        X_data = df[['AC_Units']]
        df['Predicted_Bill (INR)'] = pipeline.predict(X_data)

        # Plotting Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['AC_Units'], df['Electric_Bill (INR)'], color='blue', label='Actual Bill', s=50)
        ax.plot(df['AC_Units'], df['Predicted_Bill (INR)'], color='red', linestyle='--', linewidth=2, label='Polynomial Prediction Curve')
        ax.set_title("Actual vs Predicted Electricity Bill")
        ax.set_xlabel("AC Units Consumed")
        ax.set_ylabel("Electric Bill (INR)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

        st.pyplot(fig)
    else:
        st.error("Uploaded file must contain 'AC_Units' and 'Electric_Bill (INR)' columns.")

# ==========================================
# 2. SINGLE INPUT PREDICTION
# ==========================================
st.subheader("🔮 Predict New Bill")
ac_units = st.number_input(
    "Enter AC Electricity Consumption (Units):",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
    step=1.0
)

if st.button("Calculate Bill"):
    prediction = pipeline.predict(np.array([[ac_units]]))[0]
    st.success(f"Estimated Electric Bill for {ac_units} units: ₹{prediction:,.2f}")
