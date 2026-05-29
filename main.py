from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("Models/model.pkl", "rb"))

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.get("/predict")
def predict(
    study_hours_per_day: float,
    attendance_percentage: float,
    sleep_hours: float,
    mental_health_rating: float
):

    # Create dataframe
    data = pd.DataFrame({
        "study_hours_per_day": [study_hours_per_day],
        "attendance_percentage": [attendance_percentage],
        "sleep_hours": [sleep_hours],
        "mental_health_rating": [mental_health_rating]
    })

    # Prediction
    prediction = model.predict(data)

    return {
        "predicted_exam_score": round(float(prediction[0]), 2)
    }