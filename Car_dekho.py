import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Customizing the app's CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5; /* Light grey background */
        }
        .title {
            text-align: center; 
            color: #2a9df4; /* Light Blue color */
            font-size: 36px; 
            font-weight: bold;
            margin-bottom: 20px; /* Add space below the title */
        }
        .stSelectbox > div > div {
            font-size: 14px; /* Smaller font size for dropdown */
        }
        .predicted-price {
            color: #ff5722; /* Orange for price */
            font-weight: bold;
            font-size: 24px;
        }
        .button-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .button-container button {
            padding: 10px 20px;
            background-color: #2a9df4;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .button-container button:hover {
            background-color: #007acc;
        }
        .dropdown {
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown('<div class="title">Find the Best Price for Your Car</div>', unsafe_allow_html=True)

# Add spacing below the title
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Dropdown for City Selection
city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "hyderabad", "jaipur", "kolkata"], key="city")

# Buttons for Fuel Type
fuel_type = st.radio(
    "Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"], horizontal=True, key="fuel"
)

# Buttons for Body Type
body_type = st.radio(
    "Body Type", ["Sedan", "Hatchback", "SUV", "Minivans", "MUV", "Wagon"], horizontal=True, key="body"
)

# Buttons for Transmission Type
transmission = st.radio(
    "Transmission Type", ["Manual", "Automatic"], horizontal=True, key="transmission"
)

# Dropdowns and Inputs for Other Features
Previous_Owners = st.selectbox("Previous Owners", ["0", "1", "2", "3"], key="owners",)
engine_displacement = st.number_input(
    "Engine Displacement (cc)", min_value=793, max_value=1896, value=793, key="engine"
)
kilometers_driven = st.number_input(
    "Kilometers Driven", min_value=0, max_value=154931, value=0, key="kilometers"
)
year_of_manufacture = st.number_input(
    "Year of Manufacture", min_value=1985, max_value=2023, value=1985, key="year"
)

# Fixed value for seats
seats = 5
st.write(f"Seats (Fixed to 5): {seats}")

# Create a DataFrame from the inputs
input_data = pd.DataFrame({
    'City': [city],
    'Fuel_Type': [fuel_type],
    'Body_Type': [body_type],
    'Transmission_Type': [transmission],
    'Seats': [seats],
    'Engine_Displacement': [engine_displacement],
    'Kilometers_Driven': [kilometers_driven],
    'Year_of_Manufacture': [year_of_manufacture],
    'Previous_Owners': [Previous_Owners]
})

# Load the cleaned car data from the Excel file
car_data = pd.read_excel('C:/Users/Hp/Desktop/poorani-Projects/Project 3/Final_Cleaned_car_data.xlsx')

# Features and target variable
X = car_data.drop('Price', axis=1)
y = car_data['Price']

# Categorical columns for OneHotEncoding
categorical_features = ['City', 'Fuel_Type', 'Body_Type', 'Transmission_Type']

# Numerical columns for scaling
numerical_features = ['Seats', 'Engine_Displacement', 'Kilometers_Driven', 'Year_of_Manufacture']

# Column transformer to apply different transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create the model pipeline with Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest Regressor
])

# Train the model with the actual dataset
model.fit(X, y)

# Predict the price using the user inputs
predicted_price = model.predict(input_data)

# Display the prediction result
st.markdown(f'<div class="predicted-price">Predicted Car Price: ₹{predicted_price[0]:,.0f}</div>', unsafe_allow_html=True)