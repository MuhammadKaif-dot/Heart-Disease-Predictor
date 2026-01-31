import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict',"Model Information"])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex",['Male','Female'])
    chest_pain = st.selectbox("Chest Pain Type" , ['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0,max_value=300)
    cholesterol = st.number_input('Serum Cholestrol (mm/dl)',min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar",['<=120 mg/dl','>120 mg/dl'])
    resting_ecg = st.selectbox("Resting ECG Results",['Normal','ST-T Wave Abnormality','Left Ventricular Hypertrophy'])
    max_hr = st.number_input("Maximum Heart Rate Achieved",min_value=60,max_value=202)
    exercise_angina = st.selectbox("Exercise-Included Angina", ['Yes','No'])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0 , max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ['Up Sloping','Flat','Down Sloping'])
    
    sex = 0 if sex == 'Male' else 1
    chest_pain = ['Atypical Angina','Non-Anginal Pain','Asymptomatic',"Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == '>120 mg/dl' else 0
    resting_ecg = ['Normal','ST-T Wave Abnormality','Left Ventricular Hypertrophy'].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope =  ['Up Sloping','Flat','Down Sloping'].index(st_slope)
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    algorithm = 'XGBoostClassifier'
    modelname = 'XGB_model.pkl'
        
    def predict_heart_disease(data):
        model = pickle.load(open('XGB_model.pkl','rb'))
        prediction = model.predict(data)
        return prediction

    if st.button("Submit"):
        st.subheader('Result...')
        st.markdown('------------------------')
        result = predict_heart_disease(input_data)
        if result == 0:
            st.write("No Heart Disease Detected")
        else:
            st.write("Heart Disease Detected")
            st.markdown('------------------------')

with tab2:
    st.title("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV File",type=['csv'])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('XGB_model.pkl','rb'))
        
        input_data['Prediction'] = ''
        for i in range(len(input_data)):
            arr = input_data.iloc[i,:-1].values
            input_data['Prediction'][i] = model.predict([arr])[0]
        input_data.to_csv("PredictedHeart.csv")
        
        st.subheader("Predictions:")
        st.write(input_data)
        
    else:
        st.warning("Please Insert a correct File")
        

with tab3:
    st.subheader("Model Information")

    st.markdown("""
    **Model Used:** XGBoost Classifier  

    **Overview:**  
    This application uses an XGBoost Classifier to predict the likelihood of heart disease
    based on patient health indicators. XGBoost is an efficient and powerful gradient boosting
    algorithm known for its high performance on structured data.

    **Dataset:**  
    The model was trained on a heart disease dataset containing medical attributes such as
    age, blood pressure, cholesterol, ECG results, and exercise-related features.

    **Input Features:**  
    - Age  
    - Sex  
    - Chest Pain Type  
    - Resting Blood Pressure  
    - Cholesterol  
    - Fasting Blood Sugar  
    - Resting ECG  
    - Maximum Heart Rate  
    - Exercise-Induced Angina  
    - ST Depression (Oldpeak)  
    - ST Segment Slope  

    **Prediction Output:**  
    - 0 → No Heart Disease Detected  
    - 1 → Heart Disease Detected  

    **Disclaimer:**  
    This application is intended for educational purposes only and should not be used as a
    substitute for professional medical advice.
    """)
