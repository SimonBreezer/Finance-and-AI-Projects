import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
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
from datetime import datetime
from pandas.plotting import scatter_matrix

# Load dataset from Domo with handling for NA values
df = domo.read_dataframe('QA ML Prediction Data', query='SELECT * FROM table', na_values=["", "NA", "#N/A"])

# Data Cleaning and Conditioning
# Removing non-full cycle titles
df['remove'] = df['TITLENAME'].isin(["DRIVECLUB", "BEYOND: Two Souls PS4"]).astype(int)
df = df[df['remove'] == 0]

# Define the threshold for missing data
threshold = 0.8  # 80% of data should be present

# Calculate the percentage of non-null values for each column
non_null_ratio = df.notnull().sum() / len(df)

# Select columns where the non-null ratio is above or equal to the threshold
cols_to_keep = non_null_ratio[non_null_ratio >= threshold].index.tolist()

# Subset your DataFrame with only these columns
df_cleaned = df[cols_to_keep].copy()  # Making a copy to avoid SettingWithCopyWarning

# Impute missing values for categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Use mode for categorical data
df_cleaned.loc[:, 'PRIMARYSTATUS'] = categorical_imputer.fit_transform(df_cleaned[['PRIMARYSTATUS']]).ravel()
df_cleaned.loc[:, 'TO_CHAR(A.ACTIVITY)'] = categorical_imputer.fit_transform(df_cleaned[['TO_CHAR(A.ACTIVITY)']]).ravel()
df_cleaned.loc[:, 'Environment'] = categorical_imputer.fit_transform(df_cleaned[['Environment']]).ravel()

# Split the dataframe to Functional and Localisation
func_df = df_cleaned[df_cleaned['DEPARTMENT_C'] == "FUNCTIONALITY"]
loc_df = df_cleaned[df_cleaned['DEPARTMENT_C'] == "LOCALISATION"]

# Group by a broader set of features for Functionality
grouped_func_df = func_df.groupby(['TITLENAME', 'GENRE', 'PLATFORM', 'STUDIO', 'Size', 'Multiplayer', 'VR', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Environment', 'Sequel', 'Game_Origin_US'])['HOURS'].sum().reset_index()

# Group by a broader set of features for Localization
grouped_loc_df = loc_df.groupby(['TITLENAME', 'GENRE', 'PLATFORM', 'STUDIO', 'Size', 'Multiplayer', 'VR', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Environment', 'Sequel', 'Game_Origin_US'])['HOURS'].sum().reset_index()

# Now, 'HOURS' represents the total QA hours per unique combination of these features for each department
print("Functionality Grouped Data:")
print(grouped_func_df.head())
print("\nLocalization Grouped Data:")
print(grouped_loc_df.head())

# Define features for ML analysis
selected_features = ['GENRE', 'PLATFORM', 'STUDIO', 'Size', 'Multiplayer', 'VR', 'FIRST_RELEASE_YEAR', 'FIRST_RELEASE_MONTH', 'MULTI_PLATFORM', 'Genre_eedar', 'Gameplay_area_eedar', 'Online_eedar', 'Multiplayer_eedar', 'Combat_speed_eedar', 'Environment', 'Sequel', 'Game_Origin_US']

# Function to perform ML analysis
def perform_ml_analysis(df, name):
    X = df[selected_features]
    y = df['HOURS']
    
    # Encode categorical variables
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
    
    # Standardize the features
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.columns),
        ])
    
    # Fit and transform the data
    X_encoded = preprocessor.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    
    # Train Random Forest with Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    
    # Train XGBoost with Hyperparameter Tuning
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search_xgb.fit(X_train, y_train)
    
    best_xgb = grid_search_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    
    # Evaluate Linear Regression
    print(f"\n{name} - Linear Regression:")
    print("MSE:", mean_squared_error(y_test, y_pred_lr))
    print("R2 Score:", r2_score(y_test, y_pred_lr))
    
    # Evaluate Random Forest
    print(f"\n{name} - Random Forest:")
    print("Best parameters:", grid_search.best_params_)
    print("MSE:", mean_squared_error(y_test, y_pred_rf))
    print("R2 Score:", r2_score(y_test, y_pred_rf))
    
    # Cross-validation for Linear Regression
    cv_scores_lr = cross_val_score(model_lr, X_encoded, y, cv=5, scoring='r2')
    print(f"\n{name} - Linear Regression - Cross-validation R2 Score:", cv_scores_lr.mean())
    
    # Cross-validation for Random Forest
    cv_scores_rf = cross_val_score(best_rf, X_encoded, y, cv=5, scoring='r2')
    print(f"\n{name} - Random Forest - Cross-validation R2 Score:", cv_scores_rf.mean())
    
    # Feature Importance for Random Forest
    feature_importance = pd.DataFrame({'feature': selected_features, 'importance': best_rf.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(f"\n{name} - Feature Importances:")
    print(feature_importance)
    
# Perform analysis for Functionality
perform_ml_analysis(grouped_func_df, "Functionality")

# Perform analysis for Localization
perform_ml_analysis(grouped_loc_df, "Localization")

# Before computing the correlation matrix, encode the categorical variables
le = LabelEncoder()
for feature in selected_features:
    if grouped_func_df[feature].dtype == 'object':
        grouped_func_df[feature] = le.fit_transform(grouped_func_df[feature].astype(str))
        
# Now compute the correlation matrix
correlation_matrix = grouped_func_df[selected_features + ['HOURS']].corr()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features with Functional QA Hours')
plt.show()

# Scatter plot matrix
scatter_matrix = pd.plotting.scatter_matrix(grouped_func_df[selected_features + ['HOURS']], alpha=0.2, figsize=(20, 20), diagonal='hist')
plt.suptitle('Scatter Plot Matrix of Features vs Functional QA Hours', fontsize=20, y=0.95)
plt.show()

def plot_boxplots(df, features, target):
    n_features = len(features)
    fig, axes = plt.subplots(nrows=(n_features // 2) + (n_features % 2), ncols=2, figsize=(20, 5 * n_features))
    fig.suptitle('Box Plots of Categorical Features vs Functional QA Hours', fontsize=20)
    for i, feature in enumerate(features):
        if df[feature].dtype == 'object':
            sns.boxplot(x=feature, y=target, data=df, ax=axes[i // 2, i % 2])
            axes[i // 2, i % 2].set_title(f'{feature} vs {target}')
            axes[i // 2, i % 2].set_xticklabels(axes[i // 2, i % 2].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
# Example usage, you might want to select categorical features from selected_features
categorical_features = ['GENRE', 'PLATFORM', 'STUDIO', 'Size', 'Environment', 'Game_Origin_US']
plot_boxplots(grouped_func_df, categorical_features, 'HOURS')

def visualize_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual QA Hours')
    plt.ylabel('Predicted QA Hours')
    plt.title(f'Actual vs Predicted QA Hours ({model_name})')
    plt.show()
    
# Assuming you've stored your predictions in these variables after running the analysis
y_test_func = y_test  # From your function's output
y_pred_lr_func = y_pred_lr
y_pred_rf_func = y_pred_rf
y_pred_xgb_func = y_pred_xgb  # Assuming XGBoost was added

# Visualize for each model
visualize_predictions(y_test_func, y_pred_lr_func, 'Linear Regression - Functionality')
visualize_predictions(y_test_func, y_pred_rf_func, 'Random Forest - Functionality')
visualize_predictions(y_test_func, y_pred_xgb_func, 'XGBoost - Functionality')
