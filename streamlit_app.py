import streamlit as st
import pickle
import numpy as np

# Load trained Random Forest model
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏠 House Price Prediction App")

st.write("""
Enter the house details below to predict the Sale Price.
""")

# Input fields
OverallQual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
GrLivArea = st.number_input("Ground Living Area (sq ft)", min_value=100, value=1000)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, value=500)
GarageCars = st.number_input("Garage Capacity (cars)", min_value=0, value=1)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
YrSold = st.number_input("Year Sold", min_value=1800, max_value=2025, value=2025)
FullBath = st.number_input("Full Bathrooms", min_value=0, value=1)
HalfBath = st.number_input("Half Bathrooms", min_value=0, value=0)
OpenPorchSF = st.number_input("Open Porch Area (sq ft)", min_value=0, value=0)
EnclosedPorch = st.number_input("Enclosed Porch Area (sq ft)", min_value=0, value=0)
ScreenPorchSF = st.number_input("Screen Porch Area (sq ft)", min_value=0, value=0)

# Feature engineering calculations
HouseAge = YrSold - YearBuilt
TotalBathrooms = FullBath + 0.5 * HalfBath
TotalPorchArea = OpenPorchSF + EnclosedPorch + ScreenPorchSF

# Predict button
if st.button("Predict Sale Price"):
    X = np.array([[OverallQual, GrLivArea, TotalBsmtSF, GarageCars,
                   YearBuilt, HouseAge, TotalBathrooms, TotalPorchArea]])
    
    prediction = model.predict(X)
    
    st.success(f"Predicted Sale Price: ${prediction[0]:,.2f}")
