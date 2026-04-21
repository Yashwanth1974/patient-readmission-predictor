import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Import SMOTE
from imblearn.over_sampling import SMOTE

# 1. Setup Directories
os.makedirs("models", exist_ok=True)

# 2. Load Cleaned Data
print("Loading cleaned dataset...")
data = pd.read_csv("data/cleaned_data.csv")

# 3. Preprocessing
print("Processing features and target variable...")
data["readmitted"] = data["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

age_map = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}
data["age"] = data["age"].map(age_map)

categorical_cols = [
    "gender",
    "race",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# 4. Train-Test Split
# CRITICAL: We split BEFORE applying SMOTE to prevent data leakage!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature Scaling
print("Scaling numerical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Apply SMOTE (Balancing the Data)
print(
    f"Before SMOTE: {sum(y_train == 1)} Readmitted, {sum(y_train == 0)} Not Readmitted"
)
print("Applying SMOTE to generate synthetic readmission data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(
    f"After SMOTE: {sum(y_train_balanced == 1)} Readmitted, {sum(y_train_balanced == 0)} Not Readmitted"
)

# 7. Model Training
print("Training Gradient Boosting Model on balanced data (this will take a minute)...")
model = GradientBoostingClassifier(random_state=42)
# Notice we train on the newly balanced X and y
model.fit(X_train_balanced, y_train_balanced)

# 8. Evaluation
print("\n--- Model Evaluation ---")
predictions = model.predict(X_test_scaled)
report = classification_report(y_test, predictions)
print(report)

# 9. Save Evaluation Metrics
print("Saving evaluation metrics...")
with open("models/classification_report.txt", "w") as f:
    f.write("Gradient Boosting Classifier (with SMOTE) - Model Evaluation\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Low Risk (0)", "High Risk (1)"]
)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Hospital Readmission Confusion Matrix (SMOTE)")
plt.savefig("models/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 10. Save Artifacts
print("Saving model, scaler, and feature columns...")
pickle.dump(model, open("models/readmission_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

with open("models/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n✅ Training complete with SMOTE balancing!")
