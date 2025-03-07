# Organize the layout
import streamlit as st
import pandas as pd
import pickle

st.sidebar.title("Select One")
app_selection = st.sidebar.selectbox("Select App", ["Single Prediction", "Prediction Using Test File"])

if app_selection == "Single Prediction":
    # Load the pre-trained model
    try:
        with open('pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    # Function to show prediction result
    def show_prediction():
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

        try:
            result = pipeline.predict(sample)
            if result[0] == 1:
                st.write("🔴 An employee may **leave** the organization.")
            else:
                st.write("🟢 An employee may **stay** with the organization.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Streamlit app UI
    st.title("Predicting Employee Churn Using Machine Learning")
    
    e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
    e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
    e3 = st.slider("Number of projects assigned to", 1, 10, 5)
    e4 = st.slider("Average monthly hours worked", 50, 300, 150)
    e5 = st.slider("Time spent at the company", 1, 10, 3)
    e6 = st.radio("Whether they had a work accident", [0, 1])
    e7 = st.radio("Whether they had a promotion in the last 5 years", [0, 1])

    options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
               'RandD', 'accounting', 'hr', 'management')
    e8 = st.selectbox("Department name", options)

    options1 = ('low', 'medium', 'high')  # Fixed typo
    e9 = st.selectbox("Salary category", options1)

    if st.button("Predict"):
        show_prediction()

else:
    # Function to process CSV file
    def process_data(data):
        try:
            with open('pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            result = pipeline.predict(data)
            data['Predicted_target'] = ["🔴 Employee may **leave**" if pred == 1 else "🟢 Employee may **stay**" for pred in result]
            return data
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None

    st.title("Batch Prediction: Employee Churn")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.replace('\n', '')
            data.rename(columns={'Departments ': 'departments'}, inplace=True)
            data = data.drop_duplicates()

            processed_data = process_data(data)
            if processed_data is not None:
                st.write("Processed Data:")
                st.write(processed_data)
                processed_data.to_csv('processed_data.csv', index=False)
                st.success("Processed data saved successfully! ✅")
        except Exception as e:
            st.error(f"File processing error: {e}")

    
