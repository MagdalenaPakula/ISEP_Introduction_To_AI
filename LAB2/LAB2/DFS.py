def dfs(graph, start='S', visited=None, expanded=None):
    if visited is None:
        visited = set()
    if expanded is None:
        expanded = []
    visited.add(start)
    expanded.append(start)
    for neighbour in sorted(graph[start]):
        if neighbour not in visited:
            dfs(graph, neighbour, visited, expanded)
    return expanded, visited

graph = {
    'S': {'A', 'B', 'D'},
    'A': {'C'},
    'B': {'D'},
    'C': {'D', 'G'},
    'D': {'C', 'G'},
    'G': {}
}

expanded, result = dfs(graph)

print('Expanded nodes:', expanded)
print('DFS traversal result:', result)
