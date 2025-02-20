import streamlit as st
import pandas as pd
import joblib 

# Load models
model = joblib.load("Model.pkl")
ss = joblib.load("scaler.pkl")
encoder = joblib.load("Encoder.pkl")

# App Title
st.title("Student Performance Prediction")
st.write("A web app to predict student performance for better learning outcomes!")

# User Inputs with Default Values
hours_studied = st.number_input("How many hours did the student study per day?", min_value=0, max_value=24, value=1)
previous_score = st.number_input("Enter the student's past academic score", min_value=0, max_value=100, value=50)
sleep_hours = st.number_input("How many hours does the student sleep?", min_value=0, max_value=24, value=6)
sample_questions_practiced = st.number_input("How many sample question papers has the student practiced?", min_value=0, value=0)

# Categorical Input
extra_curricular_activities = st.selectbox('Has the student been involved in any extra curricular activities?', ('Yes', 'No'))

# Transform categorical data
extra_curricular_activities = encoder.transform([[extra_curricular_activities]])[0][0]  # Extract scalar value

# Create DataFrame
data = pd.DataFrame({
    "Hours Studied": [hours_studied],
    "Previous Scores": [previous_score],
    "Sleep Hours": [sleep_hours],
    "Sample Question Papers Practiced": [sample_questions_practiced],
    "ECA": [extra_curricular_activities]
})

# Handle missing values
data.fillna(0, inplace=True)

# Scale data
scale = ss.transform(data)

# Predict
prediction = model.predict(scale)

# Display prediction
if st.button("Predict performance"):
    if prediction[0] > 60:
        st.success(f"Student's performance prediction is {round(prediction[0],2)}")
    elif prediction[0] > 40:
        st.warning(f"Student's performance prediction is {round(prediction[0],2)}")
    else:
        st.error(f"Student's performance prediction is {round(prediction[0],2)}")
