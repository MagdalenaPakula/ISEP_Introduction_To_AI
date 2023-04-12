from collections import deque

def bfs(graph, start='S'):
    visited = set()
    expanded = []
    queue = deque([start])
    visited.add(start)
    while queue:
        vertex = queue.popleft()
        expanded.append(vertex)
        for neighbour in sorted(graph[vertex]):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return expanded, visited

graph = {
    'S': {'A', 'B', 'D'},
    'A': {'C'},
    'B': {'D'},
    'C': {'D', 'G'},
    'D': {'C', 'G'},
    'G': {}
}

expanded, result = bfs(graph)

print('Expanded nodes:', expanded)
print('BFS traversal result:', result)
