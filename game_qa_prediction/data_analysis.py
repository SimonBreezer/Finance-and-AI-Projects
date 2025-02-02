# data_analysis.py

import pandas as pd
from data_collection import generate_game_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate game data
game_data = generate_game_data(100)  # Let's use more data for better analysis

# Convert to DataFrame for easier analysis
df = pd.DataFrame(game_data)

# Basic statistics
print(df.describe())

# Correlation analysis
correlation = df.corr()
print("\nCorrelation with QA Cost:")
print(correlation['qa_cost'].sort_values(ascending=False))

# Visualize the relationship between variables and QA cost
# Scatter plots for each variable against QA cost
for column in df.columns:
    if column != 'qa_cost':
        plt.figure(figsize=(10, 6))
        plt.scatter(df[column], df['qa_cost'])
        plt.xlabel(column)
        plt.ylabel('QA Cost')
        plt.title(f'Relationship between {column} and QA Cost')
        plt.show()

# New section for data preparation for machine learning
print("\nPreparing data for machine learning:")

# Split features and target
X = df.drop('qa_cost', axis=1)
y = df['qa_cost']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Data prepared for machine learning.")
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
