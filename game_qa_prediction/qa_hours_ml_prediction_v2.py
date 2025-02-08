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

# Load the data
clean_df = pd.read_csv("Cleaned_Titles_Csv.csv", na_values=["", "NA", "#N/A"])

# Data Cleaning and Conditioning
# Removing non-full cycle titles
clean_df['remove'] = clean_df['TITLENAME'].isin(["DRIVECLUB", "BEYOND: Two Souls PS4"]).astype(int)
clean_df = clean_df[clean_df['remove'] == 0]

# Convert Dates to datetime which is easy to work with in Python
date_columns = ['Alpha_WSR', 'Beta_WSR', 'FormatQASubmission_WSR', 'ORIG_DATE']
for col in date_columns:
    clean_df[col] = pd.to_datetime(clean_df[col], format='%Y-%m-%d')

# Function to calculate months between dates
def elapsed_months(end_date, start_date):
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

# Calculate months between dates using the above function
clean_df['MONTHS_TO_ALPHA'] = clean_df.apply(lambda row: elapsed_months(row['Alpha_WSR'], row['ORIG_DATE']), axis=1)
clean_df['MONTHS_TO_BETA'] = clean_df.apply(lambda row: elapsed_months(row['Beta_WSR'], row['ORIG_DATE']), axis=1)
clean_df['MONTHS_TO_QASUBMISSION'] = clean_df.apply(lambda row: elapsed_months(row['FormatQASubmission_WSR'], row['ORIG_DATE']), axis=1)

# Split the dataframe to Functional and Localisation
func_df = clean_df[clean_df['DEPARTMENT_C'] == "FUNCTIONALITY"]
loc_df = clean_df[clean_df['DEPARTMENT_C'] == "LOCALISATION"]

# Rollup Functionality Data Frames
tot_func_hrs = func_df.groupby(['TITLENAME', 'PLATFORM', 'GENRE', 'STUDIO', 'FIRST_RELEASE_DATE', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'VR',
                                'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Sequel', 'Game_Origin_US', 'Size']).agg({
    'MONTHS_TO_RELEASE': 'max',
    'DAYS_TO_RELEASE': 'max',
    'HOURS': ['sum'],
    'is_POST_RELEASE': [lambda x: sum(x == 0), lambda x: sum(x == 1)]
}).reset_index()
tot_func_hrs.columns = ['TITLENAME', 'PLATFORM', 'GENRE', 'STUDIO', 'FIRST_RELEASE_DATE', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'VR',
                        'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Sequel', 'Game_Origin_US', 'Size',
                        'MAX_MTH_TO_REL', 'MAX_DAYS_TO_REL', 'FUNC_TOT_HRS', 'PRE_REL_HRS', 'POST_REL_HRS']

# Here, you would need to similarly handle monthly hours and other data manipulations

# Data Preparation before running models
categorical_features = ['PLATFORM', 'GENRE', 'STUDIO', 'Genre_eedar', 'Gameplay_area_eedar', 'Combat_speed_eedar', 'Game_Origin_US', 'Size']
for col in categorical_features:
    tot_func_hrs[col] = tot_func_hrs[col].astype('category')

# Remove unnecessary columns
features = tot_func_hrs.drop(columns=['TITLENAME', 'GENRE', 'FIRST_RELEASE_DATE', 'PRE_REL_HRS', 'POST_REL_HRS', 'STUDIO', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'MAX_DAYS_TO_REL'])

# Creating preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                                   colsample_bytree=1, max_depth=7))])

# Assuming you have a list of titles for testing similar to the R script
test_titles = ["Concrete Genie", "Drawn 2 Death", "Ghost of Tsushima", "Gravity Daze 2 PS4", "Kill Strain", "LocoRoco 2 Remastered", "MATTERFALL", "No Heroes Allowed! VR", "Ratchet and Clank PS4", "RIGS", "The Last Of Us 2", "TRACK LAB"]
train_data = tot_func_hrs[~tot_func_hrs['TITLENAME'].isin(test_titles)]
test_data = tot_func_hrs[tot_func_hrs['TITLENAME'].isin(test_titles)]

X_train = train_data.drop(columns=['FUNC_TOT_HRS'])
y_train = train_data['FUNC_TOT_HRS']
X_test = test_data.drop(columns=['FUNC_TOT_HRS'])
y_test = test_data['FUNC_TOT_HRS']

# Train the model
model.fit(X_train, np.log(y_train))  # Log transformation for better regression performance

# Make predictions
predictions = model.predict(X_test)
predictions = np.exp(predictions)  # Inverse transform the log predictions

# Evaluate model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions, 'Title': test_data['TITLENAME']})
plt.figure(figsize=(10, 6))
sns.barplot(data=results.melt(id_vars='Title', var_name='Type', value_name='Hours'), x='Title', y='Hours', hue='Type')
plt.title('Actual vs Predicted QA Testing Hours')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
