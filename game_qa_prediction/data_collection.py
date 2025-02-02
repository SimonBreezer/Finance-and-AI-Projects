# data_collection.py

import random

def generate_game_data(num_games):
    games_data = []
    for _ in range(num_games):
        game = {
            'quests': random.randint(10, 100),
            'levels': random.randint(5, 50),
            'characters': random.randint(20, 200),
            'items': random.randint(50, 500),
            'hours_of_gameplay': random.randint(10, 100),  # Estimated hours of gameplay
            'dialogue_lines': random.randint(500, 5000),  # Number of dialogue lines
            'cutscenes': random.randint(5, 50),  # Number of cutscenes
            'multiplayer': random.choice([True, False]),  # If the game has multiplayer
            'qa_cost': random.randint(10000, 500000)  # Simulated QA cost in dollars
        }
        games_data.append(game)
    return games_data

# Generate some sample data
sample_data = generate_game_data(10)
print(sample_data)
