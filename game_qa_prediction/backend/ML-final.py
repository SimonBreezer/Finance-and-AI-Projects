# ---------------------------------------------
# GameQA_Predictor – TC&S AI Hackathon Submission
# ---------------------------------------------
# This script trains a machine learning model to estimate monthly QA hours
# for a game title based on structured pre-release attributes.
# It uses Random Forest to predict a full QA curve (hours by month relative
# to release) without requiring 'Months_to_release' as input.
# The output includes detailed metrics (R2, MAE, MSE), feature importances,
# and saves the model and encoders for FastAPI integration.
# ---------------------------------------------

import domojupyter as domo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Load the dataset
df = domo.read_dataframe('GameQA_Predictor_v6-trimmed.xlsx', query='SELECT * FROM table')

# Drop incomplete entries
df = df.dropna(subset=[
    'HOURS', 'Months_to_release', 'Classification', 'Primary Platform',
    'Game Engine', 'Genre', 'Game World Scale', 'Gameplay Complexity',
    'Multiplayer', 'Game Modes'
])

# Encode categorical inputs
categorical_cols = [
    'Classification', 'Primary Platform', 'Game Engine', 'Genre',
    'Game World Scale', 'Gameplay Complexity', 'Multiplayer', 'Game Modes'
]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Separate static features
feature_cols = categorical_cols
X_base = df.groupby("Title")[feature_cols].first().reset_index()

# Pivot HOURS by Months_to_release
y = df.pivot_table(index="Title", columns="Months_to_release", values="HOURS")
y = y.fillna(0)

# Align features with target
df_merged = pd.merge(X_base, y, left_on="Title", right_index=True)

X = df_merged[feature_cols]
y = df_merged.drop(columns=["Title"] + feature_cols)

# Train multi-output model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"R2 Score (training): {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Optional: plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in QA Hours Prediction Model')
plt.tight_layout()
plt.show()

# Prediction function for FastAPI

def predict_full_curve(user_input_dict):
    encoded_input = []
    for col in categorical_cols:
        encoded_val = encoders[col].transform([user_input_dict[col]])[0]
        encoded_input.append(encoded_val)
    input_df = pd.DataFrame([encoded_input], columns=categorical_cols)
    prediction = model.predict(input_df)[0]
    return prediction

def compute_summary_kpis(predicted_curve):
    person_months = predicted_curve / 150
    monthly_hours = predicted_curve
    nonzero_months = monthly_hours[monthly_hours > 0]
    total_hours = monthly_hours.sum()
    total_months = (nonzero_months > 150).sum()
    peak_person_month = nonzero_months.max() / 150 if total_months > 0 else 0
    avg_person_month = total_hours / total_months / 150 if total_months > 0 else 0

    return {
        "Total_QA_Hours": round(total_hours, 0),
        "Total_Months_with_Work": int(total_months),
        "Peak_Person_Month": round(peak_person_month, 0),
        "Average_Person_Month": round(avg_person_month, 0),
        "Person_Month_Curve": list(person_months.round(0))
    }

# Save model and encoders to Domo Jupyter Workspace
output_dir = "files/ML-final-hackathon"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, f"{output_dir}/qa_model.pkl")
joblib.dump(encoders, f"{output_dir}/qa_encoders.pkl")

print("✅ Model and encoders saved to Domo Workspace for deployment.")
