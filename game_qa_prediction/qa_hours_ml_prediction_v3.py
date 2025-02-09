import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import domojupyter as domo

# Load dataset from Domo with handling for NA values
df = domo.read_dataframe('QA ML Prediction Data', query='SELECT * FROM table', na_values=["", "NA", "#N/A"])

# Display the first few rows to inspect data
print(df.head())

# Check basic info about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Define the threshold for missing data
threshold = 0.8  # 80% of data should be present

# Calculate the percentage of non-null values for each column
non_null_ratio = df.notnull().sum() / len(df)

# Select columns where the non-null ratio is above or equal to the threshold
cols_to_keep = non_null_ratio[non_null_ratio >= threshold].index.tolist()

# Subset your DataFrame with only these columns
df_cleaned = df[cols_to_keep]

# Print out the columns we're keeping and the number of rows
print("Columns kept:", cols_to_keep)
print("Number of rows after cleaning:", len(df_cleaned))

# Impute missing values for categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Use mode for categorical data
df_cleaned['PRIMARYSTATUS'] = categorical_imputer.fit_transform(df_cleaned[['PRIMARYSTATUS']]).ravel()
df_cleaned['TO_CHAR(A.ACTIVITY)'] = categorical_imputer.fit_transform(df_cleaned[['TO_CHAR(A.ACTIVITY)']]).ravel()
df_cleaned['Environment'] = categorical_imputer.fit_transform(df_cleaned[['Environment']]).ravel()

# Check if there are any remaining missing values
print(df_cleaned.isnull().sum())
