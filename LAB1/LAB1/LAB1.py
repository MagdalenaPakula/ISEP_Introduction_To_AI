from queue import PriorityQueue

def a_star(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    visited = []

    while not frontier.empty():
        current = frontier.get()
        visited.append(current)

        if current == goal:
            break

        for neighbor in graph[current]:
            new_cost = cost_so_far[current] + graph[current][neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost
                frontier.put(neighbor, priority)
                came_from[neighbor] = current

    path = []
    current = goal
    total_cost = 0
    while current != start:
        path.append(current)
        total_cost += graph[came_from[current]][current]
        current = came_from[current]
    path.append(start)
    path.reverse()
    print("Parsed Nodes:", visited)
    print("Path:", path)
    print("Total Cost:", total_cost)
    return path, total_cost


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# Graph from LAB 1:
graph = {
    'A': {'C': 4},
    'B': {'D': 4},
    'C': {'D': 1, 'G': 2},
    'D': {'C': 1, 'G': 5},
    'S': {'B': 3, 'D': 5, 'A': 2},
    'G': {}
}

start = 'S'
goal = 'G'

path, total_cost = a_star(graph, start, goal)
