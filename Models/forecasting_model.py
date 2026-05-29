import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("../Dataset/Data2.csv")

# Show first rows
print(df.head())

# Input features
X = df[[
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "mental_health_rating"
]]

# Target column
y = df["exam_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Error check
error = mean_absolute_error(y_test, predictions)

print("Model Error:", error)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")