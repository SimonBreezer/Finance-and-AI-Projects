import heapq

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic (estimated cost to goal)
        self.f = g + h  # Total cost

    def __lt__(self, other):
        return self.f < other.f  # For priority queue sorting

def heuristic(a, b):
    """Manhattan Distance heuristic function"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    """Find shortest path using A* algorithm"""
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start, None, 0, heuristic(start, goal)))

    while open_list:
        current = heapq.heappop(open_list)

        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        closed_set.add(current.position)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current.position[0] + dx, current.position[1] + dy)

            if neighbor in closed_set or grid.get(neighbor) == 1:
                continue

            new_g = current.g + 1
            heapq.heappush(open_list, Node(neighbor, current, new_g, heuristic(neighbor, goal)))

    return None  # No path found

# Example Grid (0 = walkable, 1 = obstacle)
grid = {
    (0,0): 0, (1,0): 0, (2,0): 0, (3,0): 1, (4,0): 0,
    (0,1): 1, (1,1): 0, (2,1): 1, (3,1): 0, (4,1): 0,
    (0,2): 0, (1,2): 0, (2,2): 0, (3,2): 1, (4,2): 0,
    (0,3): 0, (1,3): 1, (2,3): 0, (3,3): 0, (4,3): 0,
}

start = (0, 0)
goal = (4, 3)
path = a_star(start, goal, grid)

print("Path found:", path)
