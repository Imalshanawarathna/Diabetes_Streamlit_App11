import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import json

st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# =====================================
# Step 1 — Create dummy data + model if not exist
# =====================================
DATA_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
META_PATH = "meta.json"

if not os.path.exists(DATA_PATH):
    # Create dummy diabetes-like dataset
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, n_classes=2, random_state=42
    )
    columns = [
        "Id",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ][:8]  # because n_features=8
    df_dummy = pd.DataFrame(X, columns=columns)
    df_dummy["Outcome"] = y
    df_dummy.to_csv(DATA_PATH, index=False)
    st.info(f"Dummy dataset created at {DATA_PATH}")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    meta = {
        "results": {
            "RandomForest": {
                "accuracy": acc,
                "confusion_matrix": cm.tolist()
            }
        }
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)

    st.info("Model and meta.json created!")

# =====================================
# Step 2 — Load dataset, model, and meta
# =====================================
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

# =====================================
# Step 3 — Streamlit UI
# =====================================
st.title("Diabetes Prediction App")
st.markdown("Explore data, visualize, and predict using a trained RandomForest model.")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"],
)

if page == "Home":
    st.image(
        "https://cdn.analyticsvidhya.com/wp-content/uploads/2022/01/Diabetes-Prediction-Using-Machine-Learning.webp",
        caption="Diabetes Prediction Using Machine Learning",
        use_container_width=True,
    )

elif page == "Data Exploration":
    st.header("Data Exploration")
    st.dataframe(df.sample(10))
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

elif page == "Visualizations":
    st.header("Visualizations")
    feat = st.selectbox("Choose feature for histogram", options=[c for c in df.columns if c != "Outcome"])
    fig1 = px.histogram(df, x=feat, nbins=30)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig2 = plt.figure(figsize=(6, 5))  # smaller size here
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)

elif page == "Model Prediction":
    st.header("Model Prediction")
    input_vals = {}
    for c in df.drop(columns=["Outcome"]).columns:
        input_vals[c] = st.number_input(f"Input {c}", value=float(df[c].mean()))
    if st.button("Predict"):
        X_new = pd.DataFrame([input_vals])
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]
        st.write(f"Prediction: {pred} (1 = diabetes, 0 = no diabetes)")
        st.write(f"Probability: {prob:.4f}")

elif page == "Model Performance":
    st.header("Model Performance")
    st.json(meta["results"])
    cm = np.array(meta["results"]["RandomForest"]["confusion_matrix"])
    fig_cm = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    st.pyplot(fig_cm)
