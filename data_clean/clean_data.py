import pandas as pd
import numpy as np
import os


def clean_diabetic_data():
    input_path = "../data/diabetic_data.csv"
    output_path = "../data/cleaned_data.csv"

    print(f"Loading raw data from: {input_path}")

    # ---------------- THE FIX IS HERE ---------------- #
    # keep_default_na=False prevents pandas from destroying our "None" strings.
    # na_values=['?'] tells it to ONLY treat the '?' character as an actual NaN.
    df = pd.read_csv(input_path, na_values=["?"], keep_default_na=False)
    # ------------------------------------------------- #

    initial_rows = len(df)
    print(f"Initial dataset size: {initial_rows} rows")

    features_to_keep = [
        "age",
        "gender",
        "race",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted",
    ]
    df = df[features_to_keep]

    # Handle 'Unknown/Invalid' gender which acts as a missing value (only 3 rows)
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)

    # Now when we dropna(), it will ONLY drop rows where 'race' is '?'
    df_cleaned = df.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)

    final_rows = len(df_cleaned)
    print(f"Dropped {initial_rows - final_rows} rows containing missing/invalid data.")
    print(f"Cleaned data saved to: {output_path}")
    print(f"Final dataset size: {final_rows} rows")


if __name__ == "__main__":
    clean_diabetic_data()
