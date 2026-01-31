# Heart Disease Prediction

A web application to predict the likelihood of heart disease using patient health indicators. It uses an **XGBoost Classifier** and provides a user-friendly **Streamlit** interface.

## Features
- Predict heart disease for a single patient.
- Bulk prediction from CSV files.
- View model information and explanation.

## Project Structure
HeartDiseasePrediction/
├── XGB_model.pkl           # Trained XGBoost model
├── heart_disease_app.py    # Streamlit application
├── Heart_Disease_EDA.ipynb # Jupyter Notebook with EDA & model training
├── requirements.txt       # Python dependencies
└── README.md              # Project description

## How to Run
1. Clone the repository:
git clone https://github.com/yourusername/HeartDiseasePrediction.git
cd HeartDiseasePrediction

2. Install dependencies: 
pip install -r requirements.txt

3. streamlit run app.py
streamlit run heart_disease_app.py

