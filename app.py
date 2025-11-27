import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import os
import re

st.set_page_config(page_title="Turbofan RUL Predictor", layout="wide")

# LOAD ALL MODELS
def load_all_models():
    models = {}
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        model_file = f"model_{fd}.joblib"
        scaler_file = f"scaler_{fd}.joblib"
        feat_file = f"features_{fd}.joblib"
        q_file = f"quantiles_{fd}.joblib"

        if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(feat_file):
            models[fd] = {
                "model": joblib.load(model_file),
                "scaler": joblib.load(scaler_file),
                "features": joblib.load(feat_file),
                "q_models": joblib.load(q_file) if os.path.exists(q_file) else None
            }
    return models

all_models = load_all_models()

if len(all_models) == 0:
    st.error("No trained models found. Ensure model_FDxxx.joblib, scaler_FDxxx.joblib, features_FDxxx.joblib exist.")
    st.stop()

# FD AUTO-DETECTION LOGIC
def detect_fd(df):
    """Auto-detect FD based on number of operating regimes and sensor patterns"""

    # FD001: 1 operating condition (unique op setting distribution)
    # FD002: multiple operating conditions (wide op_setting spread)
    # FD003: similar to FD001 but different units count
    # FD004: multiple op settings + high variance

    op1_var = df["op_setting_1"].std()
    op2_var = df["op_setting_2"].std()
    num_eng = df["unit_number"].nunique()

    if op1_var < 0.02 and op2_var < 0.02:
        if num_eng <= 100:
            return "FD001"
        else:
            return "FD003"
    else:
        if num_eng <= 100:
            return "FD002"
        else:
            return "FD004"

# PREPROCESSING 
def compute_rul(df):
    max_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    return df.drop(columns=["max_cycle"])


def preprocess_data(df, fd):
    """ EXACT same feature drops as training """
    drop = ["sensor_1","sensor_5","sensor_10","sensor_16","sensor_19","op_setting_3"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    return compute_rul(df)


def engineer_features(df):
    """ EXACT same rolling features as training: only rolling mean, NOT std """
    df = df.sort_values(["unit_number","time_cycles"]).copy()

    sensors = [c for c in df.columns if c.startswith("sensor_")]
    for col in sensors:
        df[f"{col}_rm5"] = df.groupby("unit_number")[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    df["cycle_norm"] = df.groupby("unit_number")["time_cycles"].transform(lambda x: x / x.max())
    return df

# DASHBOARD UI
st.title("Turbofan Remaining Useful Life Predictor (NASA C-MAPSS)")

uploaded = st.file_uploader("Upload a turbofan TXT data file", type=["txt"])

if uploaded:
    st.write("File uploaded. Detecting dataset type...")

    cols = ['unit_number','time_cycles'] + \
           [f"op_setting_{i}" for i in range(1,4)] + \
           [f"sensor_{i}" for i in range(1,22)]

    df = pd.read_csv(uploaded, sep=r"\s+", header=None, names=cols)
    df = df.dropna(axis=1, how="all")

    detected_fd = detect_fd(df)

    st.success(f"📌 Auto-detected dataset type: **{detected_fd}**")

    if detected_fd not in all_models:
        st.error(f"No model found for {detected_fd}.")
        st.stop()

    model_pack = all_models[detected_fd]
    model = model_pack["model"]
    scaler = model_pack["scaler"]
    feature_list = model_pack["features"]
    q_models = model_pack["q_models"]

    # Engine selector
    engines = sorted(df.unit_number.unique())
    engine_id = st.selectbox("Select Engine", engines)
    sub = df[df.unit_number == engine_id]

    # Preprocess
    sub = preprocess_data(sub, detected_fd)
    sub = engineer_features(sub)

    # Validate features exist
    missing = [f for f in feature_list if f not in sub.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    # Last row only (matches training)
    X_input = sub[feature_list].iloc[-1:].values
    X_scaled = scaler.transform(X_input)

    # Prediction
    pred = float(model.predict(X_scaled)[0])

    if q_models:
        try:
            p05 = float(q_models[0.05].predict(X_scaled)[0])
            p95 = float(q_models[0.95].predict(X_scaled)[0])
        except:
            p05, p95 = None, None
    else:
        p05, p95 = None, None

    # OUTPUT
    st.header("🔍 RUL Prediction")

    if pred > 30:
        color, status = "green", "🟢 Engine Health: GOOD"
    elif pred > 15:
        color, status = "yellow", "🟡 Engine Health: WARNING"
    elif pred > 5:
        color, status = "orange", "🟠 Engine Health: HIGH RISK"
    else:
        color, status = "red", "🔴 Engine Health: CRITICAL"

    st.markdown(f"## {status}")
    st.markdown(f"### Predicted Remaining Useful Life: **{pred:.1f} cycles**")

    if p05 is not None:
        st.write(f"95% Confidence Interval: **{p05:.1f} – {p95:.1f} cycles**")
        st.progress(min(1.0, pred / 40))

    # Sensor Health
    st.subheader("🔧 Sensor Status Overview")

    sensor_cols = [c for c in sub.columns if c.startswith("sensor_")]
    last = sub.iloc[-1]

    for s in sensor_cols[:10]:
        mu, sigma = sub[s].mean(), sub[s].std()
        val = last[s]
        if abs(val - mu) < sigma:
            st.write(f"🟢 {s}: Normal")
        elif abs(val - mu) < 2 * sigma:
            st.write(f"🟡 {s}: Slight deviation")
        else:
            st.write(f"🔴 {s}: Anomalous")

