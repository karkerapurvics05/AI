import heapq
import matplotlib.pyplot as plt
import networkx as nx
graph = {
    'Andheri Station': {'DN Nagar': 3, 'Western Express Highway': 4},
    'DN Nagar': {'Juhu Circle': 4},
    'Western Express Highway': {'Juhu Circle': 5, 'JVPD': 6},
    'Juhu Circle': {'JVPD': 2, 'PVR Dynamix Juhu Mall': 5},
    'JVPD': {'PVR Dynamix Juhu Mall': 2},
    'PVR Dynamix Juhu Mall': {}
}
heuristics = {
    'Andheri Station': 10,
    'DN Nagar': 7,
    'Western Express Highway': 6,
    'Juhu Circle': 4,
    'JVPD': 2,
    'PVR Dynamix Juhu Mall': 0
}
node_positions = {
    'Andheri Station': (2, 8),
    'DN Nagar': (1, 6),
    'Western Express Highway': (3, 6),
    'Juhu Circle': (1, 4),
    'JVPD': (3, 4),
    'PVR Dynamix Juhu Mall': (2, 1)
}
def a_star_search(graph, heuristics, start, goal):
    priority_queue = [(heuristics[start], start, [start], 0)]
    visited = set()
    while priority_queue:
        f_score, current, path, g_score = heapq.heappop(priority_queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path, g_score      
        for neighbor, edge_weight in graph[current].items():
            if neighbor not in visited:
                next_g = g_score + edge_weight
                next_f = next_g + heuristics[neighbor]
                heapq.heappush(
                    priority_queue,
                    (next_f, neighbor, path + [neighbor], next_g)
                )
    return None, float('inf')
optimal_path, total_distance = a_star_search(
    graph,
    heuristics,
    'Andheri Station',
    'PVR Dynamix Juhu Mall'
)
print(f"Purvi.R.Karkera T086 :: Optimal Path Discovered: {' -> '.join(optimal_path)}")
print(f"Total Road Distance: {total_distance} km\n")
G = nx.DiGraph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)
plt.figure(figsize=(10, 8))
path_edges = list(zip(optimal_path, optimal_path[1:]))
normal_edges = [edge for edge in G.edges() if edge not in path_edges]
nx.draw_networkx_nodes(
    G,
    node_positions,
    node_size=2500,
    node_color='lightblue'
)

nx.draw_networkx_edges(
    G,
    node_positions,
    edgelist=normal_edges,
    width=1.5,
    edge_color='gray',
    arrows=True
)


nx.draw_networkx_edges(
    G,
    node_positions,
    edgelist=path_edges,
    width=3.5,
    edge_color='darkorange',
    arrows=True
)


node_labels = {
    node: f"{node}\nh(n)={heuristics[node]}"
    for node in G.nodes()
}

nx.draw_networkx_labels(
    G,
    node_positions,
    labels=node_labels,
    font_size=9,
    font_weight='bold'
)


edge_labels = nx.get_edge_attributes(G, 'weight')

formatted_edge_labels = {
    edge: f"{weight} km"
    for edge, weight in edge_labels.items()
}

nx.draw_networkx_edge_labels(
    G,
    node_positions,
    edge_labels=formatted_edge_labels,
    font_color='red'
)

plt.title(
    "Purvi.R.Karkera T086 :: A* Routing Map: Andheri Station to PVR Dynamix Juhu Mall\n"
    "(Highlighted Orange Path = Optimal Route)",
    fontsize=14
)

plt.axis('off')
plt.tight_layout()
plt.show()
