import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model_path = "wine_quality_model.pkl"
scaler_path = "wine_quality_scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# App title and description
st.title("Wine Type Prediction")
st.image("images/OIP.jpg",  use_container_width=True)
st.title("Panashe Arthur Mhonde R2111434")
st.write("""
### Predict the type of wine (Class 0 or Class 1) based on its features.
Please enter the following characteristics of the wine to get a prediction.
""")

# Input fields for features
fixed_acidity = st.number_input("Fixed Acidity (g/L)", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity (g/L)", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Citric Acid (g/L)", min_value=0.0, max_value=1.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.0, max_value=50.0, step=0.1)
chlorides = st.number_input("Chlorides (g/L)", min_value=0.0, max_value=1.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=0, max_value=100, step=1)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=0, max_value=400, step=1)
density = st.number_input("Density (g/cm³)", min_value=0.0, max_value=2.0, step=0.0001)
pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01)
sulphates = st.number_input("Sulphates (g/L)", min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input("Alcohol (% by volume)", min_value=0.0, max_value=20.0, step=0.1)

# Optional: Add Quality Feature (if applicable)
quality = st.number_input("Quality (1 to 10)", min_value=1, max_value=10, step=1)

# Create a button for prediction
if st.button("Predict Wine Type"):
    # Prepare input for the model
    features = np.array([
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, quality  # Include quality if it's a feature
    ]).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)
    st.title("🍷🍷🍷")
    # Display result
    wine_type = "Class 1 (red wine)" if prediction[0] == 1 else "Class 0 (white wine)"
    st.write(f"### Predicted Wine Type: {wine_type}")
