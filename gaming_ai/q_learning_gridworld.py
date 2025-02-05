import numpy as np
import random

# Gridworld environment (0 = empty, 1 = wall, 2 = goal)
grid = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 2],
])

# Parameters
actions = ["up", "down", "left", "right"]
q_table = np.zeros((3, 5, len(actions)))  # Q-values for each state-action pair
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

# Function to get next state
def get_next_state(state, action):
    x, y = state
    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(2, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(4, y + 1)
    return (x, y) if grid[x, y] != 1 else state  # Avoid walls

# Training the agent
for episode in range(1000):
    state = (0, 0)  # Start position
    while grid[state] != 2:  # Until reaching the goal
        if random.uniform(0, 1) < epsilon:  # Explore
            action = random.choice(actions)
        else:  # Exploit
            action = actions[np.argmax(q_table[state])]
        
        next_state = get_next_state(state, action)
        reward = 1 if grid[next_state] == 2 else -0.01  # Reward for reaching goal
        q_table[state][actions.index(action)] = (1 - alpha) * q_table[state][actions.index(action)] + \
            alpha * (reward + gamma * np.max(q_table[next_state]))
        
        state = next_state

# Print trained Q-table
print("Trained Q-Table:")
print(q_table)
