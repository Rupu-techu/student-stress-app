# student_stress_app.py
# Streamlit web app for Student Stress Level Detector
# To run: pip install streamlit joblib pandas
# Then: streamlit run student_stress_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('stress_model.pkl')  # Ensure stress_model.pkl is in the same folder

# Title and description
st.title("Student Stress Level Detector 📊🧠")
st.markdown(
    """
    Enter your daily habits and see your predicted stress level.
    This tool uses a Decision Tree model trained on student data.
    """
)

# Input fields
st.header("Your Daily Details:")
study_hours = st.number_input("Study hours per day", min_value=0, max_value=24, value=4)
sleep_hours = st.number_input("Sleep hours per night", min_value=0, max_value=24, value=6)
exercise = st.selectbox("Do you exercise?", options=["Yes", "No"])
exam_soon = st.selectbox("Do you have an exam soon?", options=["Yes", "No"])
social_time = st.selectbox(
    "Social interaction level", options=["Low", "Medium", "High"]
)

# Mapping categorical inputs to numeric values
map_ex = {"Yes": 1, "No": 0}
map_exam = {"Yes": 1, "No": 0}
map_social = {"Low": 0, "Medium": 1, "High": 2}

# Prepare input array
input_data = np.array([
    study_hours,
    sleep_hours,
    map_ex[exercise],
    map_exam[exam_soon],
    map_social[social_time]
]).reshape(1, -1)

# Prediction button
if st.button("Predict Stress Level 💡"):
    pred = model.predict(input_data)[0]
    stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    st.success(f"Your predicted stress level is: **{stress_map[pred]}**!")
    st.balloons()  # celebration

# Footer
st.markdown("---")
st.caption("Built with ❤️ in Python & Streamlit")
