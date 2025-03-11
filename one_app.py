import streamlit as st
import pandas as pd
import pickle

# Function to load the pre-trained model
@st.cache_data
def load_model():
    try:
        with open('pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'pipeline.pkl' not found! Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model once
pipeline = load_model()

# App Title
st.title("Predicting Employee Churn Using Machine Learning")

# Input features from user
e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
e3 = st.slider("Number of projects assigned to", 1, 10, 5)
e4 = st.slider("Average monthly hours worked", 50, 300, 150)
e5 = st.slider("Time spent at the company", 1, 10, 3)
e6 = st.radio("Whether they had a work accident", [0, 1])
e7 = st.radio("Whether they had a promotion in the last 5 years", [0, 1])

department_options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
                      'RandD', 'accounting', 'hr', 'management')
e8 = st.selectbox("Department name", department_options)

salary_options = ('low', 'medium', 'high')
e9 = st.selectbox("Salary category", salary_options)

# Function to make a prediction
def show_prediction(e1, e2, e3, e4, e5, e6, e7, e8, e9):
    if pipeline is None:
        st.error("🚨 Model not loaded. Prediction cannot be made.")
        return
    
    sample = pd.DataFrame({
        'satisfaction_level': [e1],
        'last_evaluation': [e2],
        'number_project': [e3],
        'average_montly_hours': [e4],
        'time_spend_company': [e5],
        'Work_accident': [e6],
        'promotion_last_5years': [e7],
        'departments': [e8],
        'salary': [e9]
    })

    st.write("📊 **Sample Input Data:**")
    st.write(sample)

    # Ensure feature names match the trained model
    required_features = pipeline.feature_names_in_ if hasattr(pipeline, "feature_names_in_") else []
    missing_features = set(required_features) - set(sample.columns)
    
    if missing_features:
        st.error(f"🚨 Missing features: {missing_features}. Check input feature names!")
        return

    # Make prediction
    try:
        result = pipeline.predict(sample)
        if result[0] == 1:
            st.write("🔴 An employee may **leave** the organization.")
        else:
            st.write("🟢 An employee may **stay** with the organization.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Predict button
if st.button("Predict"):
    show_prediction(e1, e2, e3, e4, e5, e6, e7, e8, e9)
