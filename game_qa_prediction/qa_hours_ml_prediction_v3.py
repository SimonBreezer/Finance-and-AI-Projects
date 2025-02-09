import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
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

# Drop the column with all missing data
df = df.drop(columns=['LOC_LANGUAGE'])

# Drop rows where any value is missing
df_cleaned = df.dropna()

# Check the number of rows after cleaning
print("Number of rows after cleaning:", len(df_cleaned))

# Check if there are still missing values
print(df_cleaned.isnull().sum())
