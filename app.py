# =============================
# IMPORTS
# =============================
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =============================
# LOAD MODEL
# =============================
model = joblib.load("heart_rf.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.set_page_config(page_title="CDSS", layout="centered")

st.title("🧠 Hybrid Clinical Decision Support System")

# =============================
# INPUTS
# =============================
age = st.slider("Age", 20, 90, 50)

sex = st.selectbox("Sex", [0, 1],
                   format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                  format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x])

trestbps = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
thalach = st.slider("Max Heart Rate", 60, 200, 120)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

# =============================
# FEATURE BUILDER
# =============================
def build_input():
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,

        'sex_Male': 1 if sex == 1 else 0,
        'fbs_True': 1 if fbs == 1 else 0,
        'exang_True': 1 if exang == 1 else 0,

        'cp_typical angina': 1 if cp == 0 else 0,
        'cp_atypical angina': 1 if cp == 1 else 0,
        'cp_non-anginal': 1 if cp == 2 else 0,
        'cp_asymptomatic': 1 if cp == 3 else 0,
    }

    return pd.DataFrame([data])

# =============================
# CLINICAL RULE ENGINE
# =============================
def clinical_rules():
    alerts = []

    if trestbps >= 140:
        alerts.append("⚠️ Stage 2 Hypertension (BP ≥ 140)")

    if chol >= 240:
        alerts.append("⚠️ High Cholesterol (> 240)")

    if thalach < 100:
        alerts.append("⚠️ Low Heart Rate (< 100)")

    if fbs == 1:
        alerts.append("⚠️ High Blood Sugar")

    if oldpeak > 2:
        alerts.append("⚠️ ST Depression (possible ischemia)")

    if age > 60:
        alerts.append("ℹ️ Age-related risk (> 60)")

    return alerts

# =============================
# EXPLANATION ENGINE (KEY PART)
# =============================
def generate_explanation(prob):

    reasons = []
    positives = []

    # Risk factors
    if trestbps >= 140:
        reasons.append("high blood pressure")

    if chol >= 240:
        reasons.append("elevated cholesterol")

    if thalach < 100:
        reasons.append("low heart rate")

    if fbs == 1:
        reasons.append("high blood sugar")

    if oldpeak > 2:
        reasons.append("abnormal ECG (ST depression)")

    if age > 60:
        reasons.append("advanced age")

    # Good indicators
    if trestbps < 120:
        positives.append("normal blood pressure")

    if chol < 200:
        positives.append("healthy cholesterol")

    if thalach > 150:
        positives.append("good heart rate")

    # Final explanation
    if prob > 0.7:
        return f"🔴 High risk predicted because of {', '.join(reasons)}."

    elif prob > 0.4:
        return f"🟡 Moderate risk due to {', '.join(reasons)}."

    else:
        return f"🟢 Low risk. Patient shows {', '.join(positives)}."

# =============================
# PREDICTION
# =============================
if st.button("Predict"):

    X = build_input()
    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.write(f"**Risk Score:** {prob:.2f}")

    # Risk label
    if prob > 0.7:
        st.error("🔴 HIGH RISK")
    elif prob > 0.4:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

    # =============================
    # CLINICAL ALERTS
    # =============================
    st.subheader("🚨 Clinical Alerts")

    alerts = clinical_rules()

    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.write("✅ No critical alerts")

    # =============================
    # EXPLANATION
    # =============================
    st.subheader("🧠 Explanation")

    explanation = generate_explanation(prob)
    st.info(explanation)

    # =============================
    # FINAL DECISION (FUSION)
    # =============================
    st.subheader("🧾 Final Recommendation")

    if prob > 0.7 or len(alerts) >= 3:
        st.error("Immediate medical attention recommended.")

    elif prob > 0.4:
        st.warning("Further diagnostic tests advised.")

    else:
        st.success("Maintain healthy lifestyle. Regular monitoring suggested.")