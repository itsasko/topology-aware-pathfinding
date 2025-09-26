import matplotlib
from build_graph import build_knn_graph

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from surfaces import (klein_bottle_point_cloud, crosscap_point_cloud, torus_point_cloud,
                      sphere_point_cloud)
from heuristic_search import astar, weighted_astar, greedy_bfs

def plot_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="lightcoral")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


sphere = sphere_point_cloud(num_points=500)
klein = klein_bottle_point_cloud(num_points=500)
crosscap = crosscap_point_cloud(num_points=500)
torus = torus_point_cloud(num_points=500)

# plot_point_cloud(sphere, "Sphere")
# plot_point_cloud(torus, "Torus")
# plot_point_cloud(klein, "Klein Bottle")
# plot_point_cloud(crosscap, "Crosscap")


def visualize_graph(points, G, title="Graph on Point Cloud"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="lightcoral", alpha=0.7)

    for (i, j) in G.edges():
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="gray", linewidth=0.5, alpha=0.5)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


G_sphere = build_knn_graph(sphere, k=8)
G_torus = build_knn_graph(torus, k=8)
G_klein = build_knn_graph(klein, k=8)
G_crosscap = build_knn_graph(crosscap, k=8)

# visualize_graph(sphere, G_sphere, "Sphere with k-NN Graph")
# visualize_graph(klein, G_klein, "Klein Bottle with k-NN Graph")
# visualize_graph(torus, G_torus, "Torus with k-NN Graph")
# visualize_graph(crosscap, G_crosscap, "Crosscap with k-NN Graph")


def visualize_path(points, G, path, title="Path Visualization"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="lightcoral", alpha=0.7)

    for (i, j) in G.edges():
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="gray", linewidth=0.3, alpha=0.3)

    if path:
        for u, v in zip(path[:-1], path[1:]):
            p1, p2 = points[u], points[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c="blue", linewidth=2)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    plt.show()

start, goal = 0, 100   # pick indices in your point cloud

path_a = astar(G_sphere, sphere, start, goal)
path_wa = weighted_astar(G_sphere, sphere, start, goal, w=2.0)
path_gbfs = greedy_bfs(G_sphere, sphere, start, goal)

path_klein_a = astar(G_klein, klein, start, goal)
path_klein_wa = weighted_astar(G_klein, klein, start, goal, w=2.0)
path_klein_gbfs = greedy_bfs(G_klein, klein, start, goal)

path_torus_a = astar(G_torus, torus, start, goal)
path_torus_wa = weighted_astar(G_torus, torus, start, goal, w=2.0)
path_torus_gbfs = greedy_bfs(G_torus, torus, start, goal)

path_crosscap_a = astar(G_crosscap, crosscap, start, goal)
path_crosscap_wa = weighted_astar(G_crosscap, crosscap, start, goal, w=2.0)
path_crosscap_gbfs = greedy_bfs(G_crosscap, crosscap, start, goal)

# Sphere
visualize_path(sphere, G_sphere, path_a, "A* on Sphere")
visualize_path(sphere, G_sphere, path_wa, "Weighted A* on Sphere")
visualize_path(sphere, G_sphere, path_gbfs, "Greedy BFS on Sphere")

# Klein Bottle
visualize_path(klein, G_klein, path_klein_a, "A* on Klein Bottle")
visualize_path(klein, G_klein, path_klein_wa, "Weighted A* on Klein Bottle")
visualize_path(klein, G_klein, path_klein_gbfs, "Greedy BFS on Klein Bottle")

# Torus
visualize_path(torus, G_torus, path_torus_a, "A* on Torus")
visualize_path(torus, G_torus, path_torus_wa, "Weighted A* on Torus")
visualize_path(torus, G_torus, path_torus_gbfs, "Greedy BFS on Torus")

# Crosscap
visualize_path(crosscap, G_crosscap, path_crosscap_a, "A* on Crosscap")
visualize_path(crosscap, G_crosscap, path_crosscap_wa, "Weighted A* on Crosscap")
visualize_path(crosscap, G_crosscap, path_crosscap_gbfs, "Greedy BFS on Crosscap")









