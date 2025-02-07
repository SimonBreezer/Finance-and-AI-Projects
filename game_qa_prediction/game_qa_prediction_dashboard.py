# dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
from data_collection import generate_game_data
from data_analysis import predict_qa_cost, scaler, model

def create_dashboard():
    # Generate some sample data
    game_data = generate_game_data(100)
    df = pd.DataFrame(game_data)

    # Basic statistics
    stats = df.describe()

    # Display basic statistics
    print("Basic Statistics of Game Data:")
    print(stats)

    # Plot distribution of QA costs
    plt.figure(figsize=(10, 6))
    plt.hist(df['qa_cost'], bins=20, edgecolor='black')
    plt.title('Distribution of QA Costs')
    plt.xlabel('QA Cost')
    plt.ylabel('Frequency')
    plt.show()

    # Example Prediction
    example_game = {
        'quests': 50, 
        'levels': 20, 
        'characters': 150, 
        'items': 300, 
        'hours_of_gameplay': 50, 
        'dialogue_lines': 2000, 
        'cutscenes': 15, 
        'multiplayer': True
    }
    predicted_cost = predict_qa_cost(**example_game)
    print(f"\nPredicted QA Cost for Example Game: ${predicted_cost:.2f}")

    # Visualize Example Game's Attributes
    example_df = pd.DataFrame([example_game])
    example_df_scaled = scaler.transform(example_df)
    feature_importance = pd.DataFrame({'feature': example_df.columns, 'importance': abs(model.coef_)})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance for Example Game')
    plt.xlabel('Game Attributes')
    plt.ylabel('Importance (Absolute Coefficient Value)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_dashboard()
