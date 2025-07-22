import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("clean_fuel.csv")
    return df

df = load_data()

# Encode categorical features
class_encoder = LabelEncoder()
df["VEHICLE CLASS"] = class_encoder.fit_transform(df["VEHICLE CLASS"])

transmission_encoder = LabelEncoder()
df["TRANSMISSION"] = transmission_encoder.fit_transform(df["TRANSMISSION"])

fuel_encoder = LabelEncoder()
df["FUEL"] = fuel_encoder.fit_transform(df["FUEL"])

# Define features and target
X = df[["VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "EMISSIONS"]]
y = df[["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]]

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)
# Define UI
st.title("🚗 Fuel Consumption Predictor")

vehicle_class = st.selectbox("Select Vehicle Class", list(class_encoder.classes_))
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, step=1)
transmission = st.selectbox("Transmission Type", list(transmission_encoder.classes_))
fuel = st.selectbox("Fuel Type", list(fuel_encoder.classes_))
emissions = st.number_input("Emissions (g/km)", min_value=0)

# Predict
if st.button("Predict"):
    try:
        input_data = {
            "VEHICLE CLASS": class_encoder.transform([vehicle_class])[0],
            "ENGINE SIZE": engine_size,
            "CYLINDERS": cylinders,
            "TRANSMISSION": transmission_encoder.transform([transmission])[0],
            "FUEL": fuel_encoder.transform([fuel])[0],
            "EMISSIONS": emissions
        }
        input_df = pd.DataFrame(input_data, index=[0])
        prediction = model.predict(input_df)

        st.subheader("🔍 Predicted Fuel Consumption:")
        target = ["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]
        for i, label in enumerate(target):
            st.write(f"**{label}**: {prediction[0][i]:.2f}")

    except Exception as e:
        st.error("Something went wrong. Check your inputs.")
