import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
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

# Group by a broader set of features
grouped_df = df_cleaned.groupby(['TITLENAME', 'GENRE', 'PLATFORM', 'STUDIO', 'Size', 'Multiplayer', 'VR', 'MONTHS_TO_RELEASE', 'DAYS_TO_RELEASE', 'is_POST_RELEASE', 'DEPARTMENT', 'PRIMARYSTATUS', 'RECORDTYPE', 'TO_CHAR(A.ACTIVITY)', 'BILLABLE', 'HOURTYPE', 'DEPARTMENT_C', 'is_VR', 'is_emuPS2', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'MULTI_PLATFORM', 'is_PORTED', 'PORTED', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Environment', 'Sequel', 'Game_Origin_US', 'Alpha_WSR', 'Beta_WSR', 'FormatQASubmission_WSR'])['HOURS'].sum().reset_index()

# Now, 'HOURS' represents the total QA hours per unique combination of these features
print(grouped_df.head())

# Encode categorical variables for correlation analysis
for column in grouped_df.select_dtypes(include=['object']).columns:
    if column != 'TITLENAME':  # We don't need to encode the game title for correlation
        le = LabelEncoder()
        grouped_df[column] = le.fit_transform(grouped_df[column])

# Compute correlation matrix
correlation_matrix = grouped_df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(20, 18))  # Increase size due to more features
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Features with Total QA Hours')
plt.show()

# Print correlation with HOURS for easier inspection
print(correlation_matrix['HOURS'].sort_values(ascending=False))
