import streamlit as st
import pandas as pd
import joblib 

model=joblib.load("Model.pkl")
ss=joblib.load("scaler.pkl")
encoder=joblib.load("Encoder.pkl")


st.title("Student Performance Prediction")
st.write("A web app to predict student performance")

hours_studied=st.number_input("Enter the hour studied")
previous_score=st.number_input("Enter the previous score")
sleep_hours=st.number_input("Enter the sleep hours")
sample_questions_practiced=st.number_input("Enter the number of sample question papers practiced")
extra_curricular_activities=st.selectbox('Enter if involved in extra curricular activites or not',('Yes','No'))

extra_curricular_activities=encoder.transform([[extra_curricular_activities]])

data=pd.DataFrame({"Hours Studied":hours_studied,"Previous Scores":previous_score,"Sleep Hours":sleep_hours,"Sample Question Papers Practiced":sample_questions_practiced,"ECA":extra_curricular_activities})

scale=ss.transform(data)

prediction=model.predict(scale)

if st.button("Predict performance"):
    if prediction[0]>60:
        st.success(f"Student's performance prediction is {round(prediction[0],2)}")
    elif prediction[0]>40:
        st.warning(f"Student's performance prediction is {round(prediction[0],2)}")
    else:
        st.error(f"Student's performance prediction is {round(prediction[0],2)}")