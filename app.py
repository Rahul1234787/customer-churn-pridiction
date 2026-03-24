import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Inputs
tenure = st.slider("Tenure (months)", 1, 72)
monthly_charges = st.number_input("Monthly Charges")

# Predict
if st.button("Predict"):
    input_data = np.array([[tenure, monthly_charges]])

    result = model.predict(input_data)

    if result[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will not churn ✅")

import streamlit as st

st.title("Customer Churn Prediction")
st.write("App is running successfully 🎉")
