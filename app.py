
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model, encoder, and column names
rf_model = pickle.load(open("rf_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
X_columns = pickle.load(open("X_columns.pkl", "rb"))  # To ensure column alignment

# Streamlit UI
st.title("🏡 House Rent Prediction")

# User input fields
# Arrange inputs in two columns
col1, col2 = st.columns(2)

with col1:
    seller_type = st.selectbox("Seller Type", ["Agent", "Builder", "Owner"])
    layout_type = st.selectbox("Layout Type", ["BHK", "RK"])
    property_type = st.selectbox("Property Type", ["Apartment", "Independent Floor", "Independent House", "Penthouse", "Studio Apartment"])

with col2:
    furnish_type = st.selectbox("Furnish Type", ["Furnished", "Semi-Furnished", "Unfurnished"])
    city = st.selectbox("City", ["Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"])
    
# Use unique keys for numerical inputs
area = st.number_input("Area (sqft)", min_value=100, max_value=10000, step=10, key="area_input")
bedroom = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1, key="bedroom_input")
bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, key="bathroom_input")


# Create a DataFrame with user input
new_data = pd.DataFrame({
    "seller_type": [seller_type],
    "layout_type": [layout_type],
    "property_type": [property_type],
    "furnish_type": [furnish_type],
    "city": [city],
    "area": [area],
    "bedroom": [bedroom],
    "bathroom": [bathroom]
})

# Encode categorical features using the saved encoder
encoded_features = encoder.transform(new_data[["seller_type", "layout_type", "property_type", "furnish_type", "city"]])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Combine numerical and encoded categorical data
final_input = pd.concat([new_data[["area", "bedroom", "bathroom"]], encoded_df], axis=1)

# Ensure column order matches training data
final_input = final_input.reindex(columns=X_columns, fill_value=0)

# Predict Rent
if st.button("Predict Rent Price"):
    prediction = rf_model.predict(final_input)
    st.success(f" Estimated Rent Price: ₹{prediction[0]:,.2f}")
