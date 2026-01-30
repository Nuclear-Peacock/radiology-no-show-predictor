import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier

print("--- STARTING TRAINING ---")

# 1. LOAD DATA
try:
    df = pd.read_csv("no_show_data.csv")
    print(f"✅ Loaded {len(df)} patient records.")
except:
    print("❌ Error: Could not find no_show_data.csv")
    exit()

# 2. DATA CLEANING & MATH
# Calculate 'Lead Time' (Appointment Date - Scheduled Date)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()
df['Lead_Time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Lead_Time'] = df['Lead_Time'].apply(lambda x: max(x, 0)) # Fix negatives

# Convert Text to Numbers
df['Gender_M'] = df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
df['Label'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)

# Normalize inputs (0.0 to 1.0) for the Neural Network
df['Age'] = df['Age'] / 100.0
df['Lead_Time'] = df['Lead_Time'] / 365.0

# 3. SELECT FEATURES
# We will use exactly these 8 inputs. Order matters!
features = [
    'Lead_Time', 
    'Age', 
    'Scholarship', 
    'Hipertension', 
    'Diabetes', 
    'Alcoholism', 
    'SMS_received', 
    'Gender_M'
]
X = df[features]
y = df['Label']

# 4. TRAIN NEURAL NETWORK
# A "Tiny" network: 8 Inputs -> 4 Hidden Neurons -> 1 Output
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=2000, random_state=42)
model.fit(X, y)

# 5. EXTRACT & PRINT WEIGHTS
# These 4 variables define the "Brain" of your AI
w1 = model.coefs_[0].tolist()      # Input -> Hidden Weights
b1 = model.intercepts_[0].tolist() # Hidden Biases
w2 = model.coefs_[1].tolist()      # Hidden -> Output Weights
b2 = model.intercepts_[1].tolist() # Output Bias

print("\n" + "="*50)
print("🧠 COPY THE CODE BLOCK BELOW FOR YOUR WEBSITE:")
print("="*50)
print(f"const W1 = {json.dumps(w1)};")
print(f"const B1 = {json.dumps(b1)};")
print(f"const W2 = {json.dumps(w2)};")
print(f"const B2 = {json.dumps(b2)};")
print("="*50 + "\n")
