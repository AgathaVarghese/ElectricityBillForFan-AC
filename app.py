import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Page Configuration
st.set_page_config(page_title="AC & Fan Electric Bill Predictor", layout="centered")

st.title("⚡ AC & Fan Electric Bill Predictor")
st.write("Predict electricity bill based on AC and Fan units consumed.")

# Load Trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load('model_ac_fan.pkl')

pipeline = load_model()

# ==========================================
# 1. DATASET UPLOAD & RANGE DEFINITION
# ==========================================
st.subheader("📊 Dataset Upload & Visualizer")
uploaded_file = st.file_uploader("Upload 'Day5-Christ-College-ElectricBill_AC_Fan.xlsx':", type=['xlsx', 'csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Dynamically extract min and max limits from uploaded dataset
    ac_min, ac_max = df['AC_Units'].min(), df['AC_Units'].max()
    fan_min, fan_max = df['Fan_Units'].min(), df['Fan_Units'].max()

    # Generate Actual vs Predicted Plot
    if 'Electric_Bill' in df.columns:
        df['Predicted_Bill'] = pipeline.predict(df[['AC_Units', 'Fan_Units']])
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(df.index, df['Electric_Bill'], color='blue', label='Actual Bill', s=50)
        ax.plot(df.index, df['Predicted_Bill'], color='red', linestyle='--', linewidth=2, label='Polynomial Prediction Curve')
        ax.set_title("Actual vs Predicted Electricity Bill")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Electric Bill (INR)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)
else:
    # Default range bounds from reference dataset
    ac_min, ac_max = 10, 105
    fan_min, fan_max = 20, 115

st.info(f"📌 **Allowed Ranges in Dataset:** AC Units: **{ac_min} - {ac_max}** | Fan Units: **{fan_min} - {fan_max}**")

# ==========================================
# 2. USER INPUT WITH RANGE VALIDATION
# ==========================================
st.subheader("🔮 Predict Electricity Bill")

user_ac = st.number_input("Enter AC Units Consumed:", value=50.0, step=1.0)
user_fan = st.number_input("Enter Fan Units Consumed:", value=60.0, step=1.0)

if st.button("Calculate Bill"):
    # Range Checking Logic
    is_ac_valid = ac_min <= user_ac <= ac_max
    is_fan_valid = fan_min <= user_fan <= fan_max

    if not is_ac_valid or not is_fan_valid:
        st.warning("⚠️ **Warning: Value is not within range!**")
        
        if not is_ac_valid:
            st.error(f"❌ AC Units ({user_ac}) must be within the dataset range **{ac_min} to {ac_max}**.")
        if not is_fan_valid:
            st.error(f"❌ Fan Units ({user_fan}) must be within the dataset range **{fan_min} to {fan_max}**.")
    else:
        # Input is valid, make prediction
        st.success(f"✅ Values are within valid dataset range ({ac_min}–{ac_max} for AC, {fan_min}–{fan_max} for Fan).")
        input_data = pd.DataFrame({'AC_Units': [user_ac], 'Fan_Units': [user_fan]})
        prediction = pipeline.predict(input_data)[0]
        st.metric(label="Estimated Electric Bill", value=f"₹{prediction:,.2f}")
