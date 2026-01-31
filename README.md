# Heart Disease Prediction

A web application that predicts the likelihood of heart disease using patient health indicators. The app is built with **Python**, **XGBoost**, and **Streamlit** for interactive predictions.

## Features

- Predict heart disease for a single patient or bulk data via CSV.
- Interactive web interface with user-friendly input forms.
- Fine-tuned **XGBoost classifier** achieving **89% accuracy**.
- Downloadable predictions for bulk data.
- Model information and disclaimer included.

## Technologies Used

- **Python**: Data processing and model deployment
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn & XGBoost**: Machine learning models
- **Streamlit**: Web interface
- **Pickle**: Model serialization

## How It Works

1. Load the heart disease dataset and perform EDA, cleaning, preprocessing, and visualization.
2. Train multiple ML models; select **XGBoost Classifier** for the best performance.
3. Fine-tune XGBoost to improve accuracy from 85% → 89%.
4. Save the trained model as a **pickle file** (`XGB_model.pkl`).
5. Build a **Streamlit app** with:
   - Single-patient prediction tab
   - Bulk CSV prediction tab
   - Model information tab

## Input Features

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

## Prediction Output

- `0` → No Heart Disease Detected
- `1` → Heart Disease Detected

## How to Run
1. Clone the repository:
git clone https://github.com/yourusername/HeartDiseasePrediction.git
cd HeartDiseasePrediction

2. Install dependencies: 
pip install -r requirements.txt

3. streamlit run app.py
streamlit run heart_disease_app.py
