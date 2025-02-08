import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os
import domojupyter as domo

# Load dataset from Domo
df = domo.read_dataframe('QA ML Prediction Data', query='SELECT * FROM table')

# Remove non-full cycle Titles
df = df[~df['TITLENAME'].isin(["DRIVECLUB", "BEYOND: Two Souls PS4"])]

# Convert Dates to datetime format
date_cols = ['Alpha_WSR', 'Beta_WSR', 'FormatQASubmission_WSR', 'ORIG_DATE']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Function to calculate months between dates
def elapsed_months(end_date, start_date):
    if pd.isna(end_date) or pd.isna(start_date):
        return np.nan
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

# Apply function
df['MONTHS_TO_ALPHA'] = df.apply(lambda row: elapsed_months(row['Alpha_WSR'], row['ORIG_DATE']), axis=1)
df['MONTHS_TO_BETA'] = df.apply(lambda row: elapsed_months(row['Beta_WSR'], row['ORIG_DATE']), axis=1)
df['MONTHS_TO_QASUBMISSION'] = df.apply(lambda row: elapsed_months(row['FormatQASubmission_WSR'], row['ORIG_DATE']), axis=1)

# Split the dataframe into Functional and Localization dataframes
func_df = df[df['DEPARTMENT_C'] == "FUNCTIONALITY"]
loc_df = df[df['DEPARTMENT_C'] == "LOCALISATION"]

# Aggregate functionality data
tot_func_hrs = func_df.groupby(['TITLENAME', 'PLATFORM', 'GENRE', 'STUDIO', 'FIRST_RELEASE_DATE', 'FIRST_RELEASE_YEAR',
                                'FIRST_RELEASE_MONTH', 'VR', 'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar',
                                'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Sequel', 'Game_Origin_US', 'Size']) \
    .agg(MAX_MTH_TO_REL=('MONTHS_TO_RELEASE', 'max'),
         MAX_DAYS_TO_REL=('DAYS_TO_RELEASE', 'max'),
         FUNC_TOT_HRS=('HOURS', 'sum'),
         PRE_REL_HRS=('HOURS', lambda x: x[df['is_POST_RELEASE'] == 0].sum()),
         POST_REL_HRS=('HOURS', lambda x: x[df['is_POST_RELEASE'] == 1].sum()))

# Encode categorical variables
ohe = OneHotEncoder(sparse=False, drop='first')
categorical_cols = ['PLATFORM', 'STUDIO', 'GENRE', 'Size']
ohe_data = pd.DataFrame(ohe.fit_transform(df[categorical_cols]))
ohe_data.columns = ohe.get_feature_names_out(categorical_cols)
df = pd.concat([df, ohe_data], axis=1)
df.drop(columns=categorical_cols, inplace=True)

# Select Features and Target
X = df.drop(columns=['HOURS'])  # Features
y = df['HOURS']  # Target variable

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = xgb_model.predict(X_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature Importance using SHAP
explainer = shap.Explainer(xgb_model, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test)
