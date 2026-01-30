import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="No-Show Predictor", page_icon="🏥")

st.warning("⚠️ **EDUCATIONAL DEMO ONLY** ⚠️")
st.info("This tool is for educational demonstration using public data. Do not use for clinical decisions.")

st.title("🏥 Patient No-Show Predictor")

# --- STEP 1: LOAD & PROCESS DATA ---
@st.cache_resource
def train_model():
    try:
        # Load the CSV you uploaded
        df = pd.read_csv("no_show_data.csv")
        
        # --- AUTOMATIC DATA CLEANING ---
        # 1. Calculate 'Lead Time' (Days between Scheduling and Appointment)
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()
        
        # Calculate difference in days
        df['Lead_Time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        # Fix negatives (data errors) by forcing them to 0
        df['Lead_Time'] = df['Lead_Time'].apply(lambda x: max(x, 0))
        
        # 2. Convert Text to Numbers (0 or 1)
        # Gender: M -> 1, F -> 0
        df['Gender_M'] = df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
        # Target: Yes -> 1 (They missed it), No -> 0 (They showed up)
        df['Label'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # 3. Normalize Values (Convert Age 60 to 0.6) for the AI
        df['Age_Norm'] = df['Age'].clip(upper=100) / 100.0
        df['Lead_Time_Norm'] = df['Lead_Time'].clip(upper=365) / 365.0
        
        # --- TRAINING THE AI ---
        # We pick the columns that match your CSV image
        features = ['Age_Norm', 'Lead_Time_Norm', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Gender_M']
        X = df[features]
        y = df['Label']
        
        # Create the Neural Network
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        model.fit(X, y)
        
        return model, len(df)

    except Exception as e:
        st.error(f"Error reading data: {e}")
        st.stop()

# Run the training function
model, count = train_model()
st.success(f"✅ Model trained on {count} patient records.")

# --- STEP 2: THE INTERFACE ---
st.divider()
st.write("### 🧪 Predict a Patient")
st.caption("Adjust the sliders to test the model:")

col1, col2 = st.columns(2)

with col1:
    age_input = st.slider("Patient Age", 0, 100, 30)
    lead_input = st.slider("Days Booked in Advance", 0, 120, 14)
    sms_input = st.checkbox("Received SMS Reminder?")

with col2:
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    scholarship = st.checkbox("Welfare / Scholarship")
    hypertension = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabetes")
    alcoholism = st.checkbox("Alcoholism History")

# --- STEP 3: PREDICTION LOGIC ---
if st.button("Run Prediction", type="primary"):
    # Convert user input to the same format as training data
    input_data = pd.DataFrame({
        'Age_Norm': [age_input / 100.0],
        'Lead_Time_Norm': [lead_input / 365.0],
        'Scholarship': [1 if scholarship else 0],
        'Hipertension': [1 if hypertension else 0], 
        'Diabetes': [1 if diabetes else 0],
        'Alcoholism': [1 if alcoholism else 0],
        'SMS_received': [1 if sms_input else 0],
        'Gender_M': [1 if gender == "Male" else 0]
    })
    
    # Get Probability
    probs = model.predict_proba(input_data)[0]
    risk_percent = round(probs[1] * 100, 1)
    
    st.divider()
    st.metric("No-Show Probability", f"{risk_percent}%")
    
    if risk_percent < 30:
        st.success("Analysis: **Low Risk**")
    elif risk_percent < 70:
        st.warning("Analysis: **Medium Risk**")
    else:
        st.error("Analysis: **High Risk**")
