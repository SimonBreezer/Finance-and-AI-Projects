import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

# Exclude 'TITLENAME' from correlation analysis
columns_for_correlation = grouped_df.columns.drop('TITLENAME')

# Encode categorical variables for correlation analysis
for column in columns_for_correlation:
    if grouped_df[column].dtype == 'object':  # Check if it's still a string
        le = LabelEncoder()
        try:
            grouped_df[column] = le.fit_transform(grouped_df[column])
        except:
            print(f"Error encoding column: {column}. Skipping this column.")
            columns_for_correlation = columns_for_correlation.drop(column)
            
# Compute correlation matrix with only the relevant columns
correlation_matrix = grouped_df[columns_for_correlation].corr()

# Visualize the correlation matrix
plt.figure(figsize=(20, 18))  # Increase size due to more features
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Features with Total QA Hours')
plt.show()

# Print correlation with HOURS for easier inspection
print(correlation_matrix['HOURS'].sort_values(ascending=False))

# Features to keep based on correlation (considering both positive and negative values)
selected_features = ['FIRST_RELEASE_YEAR', 'FormatQASubmission_WSR', 'Sequel', 'Beta_WSR', 'Alpha_WSR', 'Environment', 'Combat_speed_eedar', 'DAYS_TO_RELEASE', 'MONTHS_TO_RELEASE', 'MULTI_PLATFORM', 'GENRE', 'Online_eedar', 'Genre_eedar', 'Multiplayer', 'PORTED', 'Multiplayer_eedar', 'is_POST_RELEASE', 'PRIMARYSTATUS', 'is_VR', 'FIRST_RELEASE_MONTH', 'VR', 'PLATFORM', 'Gameplay_area_eedar', 'Size']

# Create features and target sets
X = grouped_df[selected_features]
y = grouped_df['HOURS']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Identify categorical columns
categorical_features = X.select_dtypes(include=['int64']).columns  # Assuming all encoded features are now integers

# Column transformer to apply OneHotEncoder to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# Fit and transform the data
X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Example with Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Example with Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Evaluate Linear Regression
print("Linear Regression - MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression - R2 Score:", r2_score(y_test, y_pred_lr))

# Evaluate Random Forest
print("Random Forest - MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest - R2 Score:", r2_score(y_test, y_pred_rf))
