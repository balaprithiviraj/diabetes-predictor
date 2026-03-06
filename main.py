from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# 1. Create the FastAPI app instance
app = FastAPI()

# 2. Allow your website to talk to this backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the model file you uploaded to GitHub
model = joblib.load("diabetes_model.joblib")

# 4. Define what data the website will send
class PatientData(BaseModel):
    glucose: float
    bmi: float
    age: int

# 5. The "Predict" route
@app.get("/")
def home():
    return {"status": "Backend is running!"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input into the format the model expects
    df = pd.DataFrame([[data.glucose, data.bmi, data.age]], columns=["Glucose", "BMI", "Age"])
    prediction = model.predict(df)[0]
    return {"result": "At Risk" if int(prediction) == 1 else "Healthy"}
