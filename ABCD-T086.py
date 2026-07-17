print("Purvi.R.Karkera T086 :: ")
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 2, 'E': 5},
    'C': {'F': 1, 'G': 3},
    'D': {'H': 4},
    'E': {'H': 2},
    'F': {'I': 5},
    'G': {'I': 2},
    'H': {'Goal': 3},
    'I': {'Goal': 2},
    'Goal': {}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 5,
    'E': 3,
    'F': 4,
    'G': 2,
    'H': 2,
    'I': 1,
    'Goal': 0
}

from queue import PriorityQueue

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 2, 'E': 5},
    'C': {'F': 1, 'G': 3},
    'D': {'H': 4},
    'E': {'H': 2},
    'F': {'I': 5},
    'G': {'I': 2},
    'H': {'Goal': 3},
    'I': {'Goal': 2},
    'Goal': {}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 5,
    'E': 3,
    'F': 4,
    'G': 2,
    'H': 2,
    'I': 1,
    'Goal': 0
}

start = 'A'
goal = 'Goal'

pq = PriorityQueue()
pq.put((heuristic[start], 0, [start]))   

visited = set()

while not pq.empty():

    f, g, path = pq.get()
    node = path[-1]

    if node == goal:
        print("Best Path:", " -> ".join(path))
        print("Total Cost =", g)
        break

    if node in visited:
        continue

    visited.add(node)

    for neighbour, cost in graph[node].items():
        new_g = g + cost
        new_f = new_g + heuristic[neighbour]

        print(path + [neighbour],
              "g =", new_g,
              "h =", heuristic[neighbour],
              "f =", new_f)

        pq.put((new_f, new_g, path + [neighbour]))
