import matplotlib.pyplot as plt
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


coords = {
    'Andheri Station': (2, 6),
    'DN Nagar': (1, 4.5),
    'Western Express Highway': (3, 4.5),
    'Juhu Circle': (1, 3),
    'JVPD': (3, 3),
    'PVR Dynamix Juhu Mall': (2, 1)
}

def rbfs_search(start, goal):


    success, path, cost, _ = rbfs(
        start,
        goal,
        g=0,
        f_limit=float('inf'),
        path=[start]
    )

    return path, cost


def rbfs(node, goal, g, f_limit, path):

    if node == goal:
        return True, path, g, g

    neighbors = graph[node]

    if not neighbors:
        return False, [], 0, float('inf')

    successors = []

    for neighbor, distance in neighbors.items():

        if neighbor not in path:

            next_g = g + distance
            next_f = max(
                next_g + heuristics[neighbor],
                g + heuristics[node]
            )

            successors.append([next_f, neighbor, next_g])

    if not successors:
        return False, [], 0, float('inf')

    while True:

        successors.sort(key=lambda x: x[0])

        best = successors[0]

        if best[0] > f_limit:
            return False, [], 0, best[0]

        alternative_f = (
            successors[1][0]
            if len(successors) > 1
            else float('inf')
        )

        success, result_path, total_g, returned_f = rbfs(
            best[1],
            goal,
            best[2],
            min(f_limit, alternative_f),
            path + [best[1]]
        )

        best[0] = returned_f

        if success:
            return True, result_path, total_g, returned_f

path, total_dist = rbfs_search(
    'Andheri Station',
    'PVR Dynamix Juhu Mall'
)

print(
    f"Purvi.R.Karkera T086 :: Optimal RBFS Path: {' -> '.join(path)} ({total_dist} km)"
)

plt.figure(figsize=(8, 7))


for node, neighbors in graph.items():

    x1, y1 = coords[node]

    for neighbor, dist in neighbors.items():

        x2, y2 = coords[neighbor]

        is_path = (
            node in path and
            neighbor in path and
            path.index(neighbor) ==
            path.index(node) + 1
        )

        color, width = (
            ('#2ecc71', 3)
            if is_path
            else ('#bdc3c7', 1.5)
        )

        plt.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width,
            zorder=1
        )

        plt.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            f"{dist}km",
            color='red',
            fontsize=9,
            ha='center'
        )

for node, (x, y) in coords.items():

    color = (
        '#f1c40f'
        if node in path
        else '#3498db'
    )

    plt.scatter(
        x,
        y,
        color=color,
        s=700,
        zorder=2
    )

    plt.text(
        x,
        y,
        f"{node}\nh={heuristics[node]}",
        ha='center',
        va='center',
        color='white',
        fontsize=8,
        fontweight='bold'
    )

plt.title(
    f"Purvi.R.Karkera T086 :: RBFS Route Map: Andheri Station to PVR Dynamix Juhu Mall (Total: {total_dist} km)",
    fontsize=12,
    fontweight='bold'
)

plt.axis('off')
plt.show()
