import streamlit as st
import pandas as pd
import joblib 

model=joblib.load("Model.pkl")
ss=joblib.load("scaler.pkl")
encoder=joblib.load("Encoder.pkl")


st.title("Student Performance Prediction")
st.write("A web app to predict student performance for better learning outcome!")

hours_studied=st.number_input("How many hours did the student study per day?",max_value=24)
previous_score=st.number_input("Enter the student's past academic score",value=None,placeholder=" ")
sleep_hours=st.number_input("How many hours does the student sleep?",max_value=24)
sample_questions_practiced=st.number_input("How many sample question papers has the student practiced?",value=0)
extra_curricular_activities=st.selectbox('Has the student been involved in any extra curricular activites?',('Yes','No'))

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

