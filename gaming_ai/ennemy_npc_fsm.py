class NPC:
    def __init__(self):
        self.state = "Idle"

    def update(self, player_distance):
        if self.state == "Idle":
            if player_distance < 10:
                self.state = "Chasing"
        
        elif self.state == "Chasing":
            if player_distance < 3:
                self.state = "Attacking"
            elif player_distance > 15:
                self.state = "Idle"
        
        elif self.state == "Attacking":
            if player_distance > 5:
                self.state = "Chasing"

    def get_state(self):
        return self.state


# Simulating an NPC reacting to player movement
npc = NPC()

distances = [15, 8, 4, 2, 6, 12, 18]  # Simulated player distances
for distance in distances:
    npc.update(distance)
    print(f"Player distance: {distance}, NPC State: {npc.get_state()}")
