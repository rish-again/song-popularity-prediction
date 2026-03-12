import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Load dataset for visualizations
df = pd.read_csv("data/songs.csv")

# Remove unwanted column if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# -----------------------------
# App Title
# -----------------------------
st.title("🎵 Song Popularity Prediction")

st.write("Enter song features to predict popularity.")

# -----------------------------
# Random Song Generator
# -----------------------------
if st.button("🎲 Generate Random Song"):
    random_values = {}

    for col in feature_columns:
        if col in df.columns:
            random_values[col] = float(df[col].sample())

    st.session_state["random_song"] = random_values


# -----------------------------
# User Inputs
# -----------------------------
input_data = {}

for feature in feature_columns:

    default_value = 0.5

    if "random_song" in st.session_state:
        if feature in st.session_state["random_song"]:
            default_value = st.session_state["random_song"][feature]

    if feature == "tempo":
        value = st.slider(feature, 50.0, 200.0, float(default_value))

    elif feature == "loudness":
        value = st.slider(feature, -60.0, 0.0, float(default_value))

    elif feature == "duration_ms":
        value = st.number_input(feature, value=int(default_value))

    elif feature in ["key", "mode", "time_signature"]:
        value = st.number_input(feature, value=int(default_value))

    else:
        value = st.slider(feature, 0.0, 1.0, float(default_value))

    input_data[feature] = value


# -----------------------------
# Convert to dataframe
# -----------------------------
input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Popularity"):

    scaled_features = scaler.transform(input_df)

    prediction = model.predict(scaled_features)[0]

    st.subheader(f"🎧 Predicted Popularity Score: {prediction:.2f}")

    # -----------------------------
    # Popularity Category
    # -----------------------------
    if prediction < 40:
        st.error("📉 Low Popularity")

    elif prediction < 70:
        st.warning("⭐ Medium Popularity")

    else:
        st.success("🔥 HIT SONG Potential!")


# =============================
# Feature Importance
# =============================
st.header("📊 Feature Importance")

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(by="importance", ascending=False)

    fig, ax = plt.subplots()

    sns.barplot(data=importance, x="importance", y="feature", ax=ax)

    st.pyplot(fig)


# =============================
# Dataset Visualizations
# =============================
st.header("📈 Dataset Insights")

# Popularity distribution
fig1, ax1 = plt.subplots()
sns.histplot(df["popularity"], bins=30, ax=ax1)

ax1.set_title("Popularity Distribution")

st.pyplot(fig1)

# Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(10,6))

corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm", ax=ax2)

ax2.set_title("Feature Correlation")

st.pyplot(fig2)


