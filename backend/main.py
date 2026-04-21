from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import sqlite3

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open("../models/readmission_model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

with open("../models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ---------------- FASTAPI & CORS ---------------- #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- SCHEMAS ---------------- #
class AuthData(BaseModel):
    username: str
    password: str


class PatientData(BaseModel):
    age: str
    gender: str
    race: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    # Reverted Complex Metrics
    max_glu_serum: str
    A1Cresult: str
    insulin: str
    change: str
    diabetesMed: str


# ---------------- AUTHENTICATION ---------------- #


@app.post("/register/")
def register(data: AuthData):
    try:
        conn = sqlite3.connect("../database/patients.db")
        cursor = conn.cursor()
        # Check if user exists
        cursor.execute("SELECT * FROM staff WHERE username=?", (data.username,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Username already exists")

        cursor.execute(
            "INSERT INTO staff(username, password) VALUES(?, ?)",
            (data.username, data.password),
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "Account created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login/")
def login(data: AuthData):
    conn = sqlite3.connect("../database/patients.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM staff WHERE username=? AND password=?",
        (data.username, data.password),
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ---------------- PREDICTION ---------------- #


@app.post("/predict/")
def predict(data: PatientData):
    input_data = pd.DataFrame([data.dict()])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_data)

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    # Save to DB
    conn = sqlite3.connect("../database/patients.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions(age, gender, race, time_in_hospital, prediction, probability) VALUES(?,?,?,?,?,?)",
        (
            data.age,
            data.gender,
            data.race,
            data.time_in_hospital,
            prediction,
            probability,
        ),
    )
    conn.commit()
    conn.close()

    return {"prediction": prediction, "probability": probability}
