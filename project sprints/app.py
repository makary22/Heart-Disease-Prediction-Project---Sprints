import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ===== Load Models =====
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "SVC": pickle.load(open("svc_model.pkl", "rb")),
    "K-Means (Unsupervised)": pickle.load(open("kmeans_model.pkl", "rb")),
    "Hierarchical Clustering": pickle.load(open("hierarchical_model.pkl", "rb"))
}

st.title("💓 Heart Disease Prediction & Clustering App")

# ===== User Input =====
st.sidebar.header("Enter Patient Data")
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
thalach = st.sidebar.number_input("Max HR", 60, 220, 150)
oldpeak = st.sidebar.number_input("Oldpeak", -2.0, 6.0, 1.0)

# input as dataframe
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, thalach, oldpeak]],
                    columns=["age","sex","cp","trestbps","chol","thalach","oldpeak"])

# ===== Choose Model =====
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_choice]

# ===== Prediction =====
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]

        if model_choice in ["K-Means (Unsupervised)", "Hierarchical Clustering"]:
            st.info(f"🔎 Cluster Assigned: {prediction}")
        else:
            if prediction == 0:
                st.success("✅ No Heart Disease")
            else:
                st.error(f"⚠️ Heart Disease Detected (Severity Level: {prediction})")


    except Exception as e:
        st.error(f"Error while predicting: {e}")

# ===== Visualization Example =====
st.subheader("📊 Heart Disease Trends (Demo)")
try:
    sample_data = pd.read_csv("heart.csv")  
    fig, ax = plt.subplots()
    sample_data["age"].hist(ax=ax, bins=20)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
except:
    st.warning("⚠️ Dataset 'heart.csv' not found. Add it to enable visualization.")
