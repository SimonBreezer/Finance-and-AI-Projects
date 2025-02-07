class Node:
    """Base class for all behavior tree nodes."""
    def run(self):
        pass

class Selector(Node):
    """Runs children until one succeeds."""
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if child.run():
                return True
        return False

class Sequence(Node):
    """Runs children in order, failing if any fail."""
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if not child.run():
                return False
        return True

class Condition(Node):
    """Returns True/False based on a condition."""
    def __init__(self, condition_func):
        self.condition_func = condition_func

    def run(self):
        return self.condition_func()

class Action(Node):
    """Executes an action and returns True (success)."""
    def __init__(self, action_func):
        self.action_func = action_func

    def run(self):
        self.action_func()
        return True

# Game AI Logic
player_visible = False
enemy_close = False

def check_player_visible():
    return player_visible

def check_enemy_close():
    return enemy_close

def patrol():
    print("NPC is patrolling...")

def investigate():
    print("NPC is investigating a noise...")

def attack():
    print("NPC is attacking the player!")

# Behavior Tree Setup
bt = Selector([
    Sequence([
        Condition(check_enemy_close),
        Action(attack)
    ]),
    Sequence([
        Condition(check_player_visible),
        Action(investigate)
    ]),
    Action(patrol)
])

# Simulating Behavior
for state in [(False, False), (True, False), (True, True)]:
    player_visible, enemy_close = state
    print(f"\nGame State: player_visible={player_visible}, enemy_close={enemy_close}")
    bt.run()
