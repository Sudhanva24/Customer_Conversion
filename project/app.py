import streamlit as st
import numpy as np
import xgboost as xgb
import pickle
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
# Load your trained XGBoost model
# Replace 'model.pkl' with the actual path to your saved model
try:
    model_path = os.path.join(os.path.dirname(__file__), '/Users/sudhanvasavyasachi/Desktop/Projects/Customer_Conversion/project/xgboost_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'xgboost_model.pkl' exists in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


# # Streamlit app
# st.title("Customer Conversion Prediction")

# st.header("Enter the Features")
# f0 = st.number_input("Feature f0", value=0.0)
# f1 = st.number_input("Feature f1", value=0.0)
# f2 = st.number_input("Feature f2", value=0.0)
# f3 = st.number_input("Feature f3", value=0.0)
# f4 = st.number_input("Feature f4", value=0.0)
# f5 = st.number_input("Feature f5", value=0.0)
# f6 = st.number_input("Feature f6", value=0.0)
# f7 = st.number_input("Feature f7", value=0.0)
# f8 = st.number_input("Feature f8", value=0.0)
# f9 = st.number_input("Feature f9", value=0.0)
# f10 = st.number_input("Feature f10", value=0.0)
# f11 = st.number_input("Feature f11", value=0.0)
# treatment = st.checkbox("In Treatment Group?", value=False)
# exposure = st.checkbox("Was Exposed?", value=False)

# # Convert to 0/1
# treatment = 1 if treatment else 0
# exposure = 1 if exposure else 0

# # Predict button
# if st.button("Predict"):
#     # Prepare input data
#     input_data = np.array([[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, treatment, exposure]])
    
#     # Make prediction
#     prediction = model.predict(input_data)
#     probability = model.predict_proba(input_data)[0][1]  # Class 1
#     prediction_label = "Convert" if probability >=0.5 else "Not Convert"
    
#     # Display result
#     st.subheader("Prediction")
#     st.write(f"The customer will: **{prediction_label}**")
#     st.write(f"Conversion Probability: **{probability:.2%}**")

#     explainer = shap.Explainer(model)
#     shap_values = explainer(input_data)
#     num_features = input_data.shape[1]
#     # Display SHAP values waterfall plot
#     st.subheader("SHAP Values Waterfall Plot")
#     shap.initjs()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     shap.plots.waterfall(shap_values[0], show=False,max_display=num_features)
#     st.pyplot(fig)
st.title("Customer Conversion Prediction")

st.header("Enter the Features")
f0 = st.number_input("Feature f0", value=0.0)
f1 = st.number_input("Feature f1", value=0.0)
f2 = st.number_input("Feature f2", value=0.0)
f3 = st.number_input("Feature f3", value=0.0)
f4 = st.number_input("Feature f4", value=0.0)
f5 = st.number_input("Feature f5", value=0.0)
f6 = st.number_input("Feature f6", value=0.0)
f7 = st.number_input("Feature f7", value=0.0)
f8 = st.number_input("Feature f8", value=0.0)
f9 = st.number_input("Feature f9", value=0.0)
f10 = st.number_input("Feature f10", value=0.0)
f11 = st.number_input("Feature f11", value=0.0)
treatment = st.checkbox("In Treatment Group?", value=False)
exposure = st.checkbox("Was Exposed?", value=False)

# Convert checkboxes to 0/1
treatment = 1 if treatment else 0
exposure = 1 if exposure else 0

# Predict button
if st.button("Predict"):
    # Create DataFrame with feature names
    input_data = pd.DataFrame([[
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, treatment, exposure
    ]], columns=[
        "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "treatment", "exposure"
    ])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of class 1
    prediction_label = "Convert" if probability >= 0.5 else "Not Convert"
    
    # Display result
    st.subheader("Prediction")
    st.write(f"The customer will: **{prediction_label}**")
    st.write(f"Conversion Probability: **{probability:.2%}**")

    # SHAP explanation
    st.subheader("SHAP Values Waterfall Plot")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    shap.initjs()
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False, max_display=len(input_data.columns))
    st.pyplot(fig)