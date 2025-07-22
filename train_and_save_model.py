import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and clean data
data = pd.read_csv("clean_fuel.csv")
data.drop_duplicates(inplace=True)

# Encode categorical variables
class_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

data["VEHICLE CLASS"] = class_encoder.fit_transform(data["VEHICLE CLASS"])
data["TRANSMISSION"] = transmission_encoder.fit_transform(data["TRANSMISSION"])
data["FUEL"] = fuel_encoder.fit_transform(data["FUEL"])

# Define features and targets
features = ["VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "EMISSIONS"]
target = ["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]

X = data[features]
y = data[target]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'fuel_model.pkl')
joblib.dump(class_encoder, 'class_encoder.pkl')
joblib.dump(transmission_encoder, 'transmission_encoder.pkl')
joblib.dump(fuel_encoder, 'fuel_encoder.pkl')
