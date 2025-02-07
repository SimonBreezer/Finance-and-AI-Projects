# data_analysis.py

import pandas as pd
from data_collection import generate_game_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# New function for visualizing feature importance
def visualize_feature_importance(feature_importance):
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance in Predicting QA Costs')
    plt.xlabel('Game Attributes')
    plt.ylabel('Importance (Absolute Coefficient Value)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

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

# New section for building and evaluating the model
print("\nBuilding and evaluating the Linear Regression model:")

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual QA Cost')
plt.ylabel('Predicted QA Cost')
plt.title('Actual vs Predicted QA Costs')
plt.show()

# Feature importance (coefficients)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Call the visualization function
visualize_feature_importance(feature_importance)

def predict_qa_cost(quests, levels, characters, items, hours_of_gameplay, dialogue_lines, cutscenes, multiplayer):
    # Create a list with the input in the same order as the features in X
    new_game_features = [[quests, levels, characters, items, hours_of_gameplay, dialogue_lines, cutscenes, int(multiplayer)]]
    
    # Scale the features using the same scaler we used for training data
    new_game_scaled = scaler.transform(new_game_features)
    
    # Predict the QA cost
    predicted_qa_cost = model.predict(new_game_scaled)
    
    # Since the prediction is an array, we return the first (and only) element
    return predicted_qa_cost[0]

# Example usage
# Let's say we want to predict for a game with:
# 50 quests, 20 levels, 150 characters, 300 items, 50 hours of gameplay, 2000 dialogue lines, 15 cutscenes, and it's multiplayer
specific_game_cost = predict_qa_cost(50, 20, 150, 300, 50, 2000, 15, True)
print(f"Predicted QA Cost for the example game: ${specific_game_cost:.2f}")
# You can now call this function with any set of game variables you want to predict for
