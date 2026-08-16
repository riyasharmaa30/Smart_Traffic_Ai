import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load traffic dataset
data_path = "data/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Convert date/time column
df["date_time"] = pd.to_datetime(df["date_time"])

# Create useful time-based features
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.dayofweek

# Convert weather-related categorical columns
df["holiday"] = df["holiday"].fillna("None")
df["weather_main"] = df["weather_main"].fillna("Unknown")

# One-hot encode categorical features
df = pd.get_dummies(
    df,
    columns=["holiday", "weather_main"],
    drop_first=True
)

# Remove rows with missing values
df = df.dropna()

# Select features
features = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all"
]

# Add encoded weather/holiday columns
extra_features = [
    col for col in df.columns
    if col.startswith("holiday_") or col.startswith("weather_main_")
]

features += extra_features

X = df[features]
y = df["traffic_volume"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Smart Traffic AI...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== SMART TRAFFIC AI RESULTS =====")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 4))

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and feature names
joblib.dump(model, "model/traffic_model.pkl")
joblib.dump(features, "model/features.pkl")

print("\nModel saved successfully!")
print("Location: model/traffic_model.pkl")
print("Smart Traffic AI training completed!")
