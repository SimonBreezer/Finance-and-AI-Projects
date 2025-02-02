# data_analysis.py

import pandas as pd
from data_collection import generate_game_data

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
import matplotlib.pyplot as plt

# Scatter plots for each variable against QA cost
for column in df.columns:
    if column != 'qa_cost':
        plt.figure(figsize=(10, 6))
        plt.scatter(df[column], df['qa_cost'])
        plt.xlabel(column)
        plt.ylabel('QA Cost')
        plt.title(f'Relationship between {column} and QA Cost')
        plt.show()
