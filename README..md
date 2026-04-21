Hospital Readmission Prediction System

This project predicts whether a hospital patient will be readmitted within 30 days using machine learning.

Dataset

Diabetes 130-US hospitals dataset (~101k patient records)

Features Used

age

gender

race

time_in_hospital

num_lab_procedures

num_procedures

num_medications

number_outpatient

number_emergency

number_inpatient

number_diagnoses

max_glu_serum

A1Cresult

insulin

change

diabetesMed

Model

Gradient Boosting Classifier (with SMOTE balancing)

Tech Stack

Python
Scikit-learn
FastAPI
HTML/Tailwind CSS
SQLite

How to Run

Create Database
python create_database.py

Clean Data
python clean_data.py

Train Model
python train_model.py

Start API
python -m uvicorn backend.main:app --reload

Start UI
Open index.html in any web browser