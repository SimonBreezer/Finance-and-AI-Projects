class EnemyAI:
    def __init__(self, player_distance, player_health):
        self.player_distance = player_distance
        self.player_health = player_health

    def decide_action(self):
        if self.player_distance < 5:  # Player is close
            if self.player_health < 30:
                return "Attack"
            else:
                return "Defend"
        else:  # Player is far
            return "Patrol"

# Example usage
enemy1 = EnemyAI(player_distance=3, player_health=20)
print(f"Enemy action: {enemy1.decide_action()}")  # Expected: Attack

enemy2 = EnemyAI(player_distance=3, player_health=80)
print(f"Enemy action: {enemy2.decide_action()}")  # Expected: Defend

enemy3 = EnemyAI(player_distance=10, player_health=50)
print(f"Enemy action: {enemy3.decide_action()}")  # Expected: Patrol
