import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import csv

# === Load Artifacts ===
model = joblib.load("models/XGBoost.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# === Title ===
st.title("🔍 Real-Time Fraud Detection")
st.markdown("Check if a transaction is fraudulent **before** it occurs.")

# === Inputs ===
step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
type_option = st.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'])
amount = st.number_input("Transaction Amount", min_value=0.0, value=10000.0)
oldbalanceOrg = st.number_input("Sender's Old Balance", min_value=0.0, value=50000.0)
nameDest = st.text_input("Receiver Name (e.g. M123 or C123)", value="M123456789")

# === Feature Engineering ===
remainingBalance = oldbalanceOrg - amount
receiver_is_merchant = 1 if nameDest.startswith("M") else 0

# STEP BIN LOGIC 
step_bin= None
def step_to_bin(step):
    if step <= 74:
        return 0
    elif step <= 148:
        return 1
    elif step <= 222:
        return 2
    elif step <= 296:
        return 3
    elif step <= 370:
        return 4
    elif step <= 444:
        return 5
    elif step <= 518:
        return 6
    elif step <= 592:
        return 7
    elif step <= 666:
        return 8
    else:
        return 9

# Type encoding (match your training encoding)
type_dict = {'CASH_OUT': 4, 'TRANSFER': 1, 'PAYMENT': 2, 'DEBIT': 3}
type_encoded = type_dict[type_option]

# Receiver type
if nameDest.startswith('M'):
    is_merchant = 1
elif nameDest.startswith('C'):
    is_merchant = 0
else:
    is_merchant = st.selectbox(
        "Is the Receiver a Merchant?",
        [0, 1],
        format_func=lambda x: "No (Customer)" if x == 0 else "Yes (Merchant)"
    )
amount_ratio_old = amount / (oldbalanceOrg + 1e-5)
#amount_ratio_new = amount / (newbalanceOrig + 1e-5)


# === Assemble Feature Vector ===
input_data = {
    'step_bin': step_bin,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'type_encoded': type_encoded,
    'remainingBalance_clipped': remainingBalance,  # same as used in training
    'receiver_is_merchant': receiver_is_merchant,
    'amount_ratio_old': amount_ratio_old,
    #'amount_ratio_new': amount_ratio_new
}

input_df = pd.DataFrame([input_data])

# Add missing columns (set 0)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# === Scale Numeric Columns ===
num_cols = ['step_bin','amount', 'oldbalanceOrg', 'type_encoded','remainingBalance_clipped', 'amount_ratio_old']
#bin_cols = ['receiver_is_merchant']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# === Predict ===
if st.button("🚨 Predict Fraud"):
    probability = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.1

    # Base prediction from model
    prediction = 1 if probability > threshold else 0

    # Rule-based overrides
    if amount_ratio_old > 0.9 or amount > oldbalanceOrg:
        prediction = 1
        probability = max(probability, 0.90)  # Boost probability if rule triggers

    # Show result
    st.write("Scaled Input Features:", input_df)

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction. (Fraud Probability: {probability:.2%})")

    # === Log to CSV ===
    with open("log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            step, amount, oldbalanceOrg, nameDest,
            prediction, f"{probability:.4f}"
        ])
