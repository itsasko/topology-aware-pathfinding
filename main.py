import matplotlib
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from build_graph import build_knn_graph

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from surfaces import (klein_bottle_point_cloud, crosscap_point_cloud, torus_point_cloud,
                      sphere_point_cloud)


def plot_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="lightcoral")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


sphere = sphere_point_cloud(num_points=5000)
klein = klein_bottle_point_cloud(num_points=5000)
crosscap = crosscap_point_cloud(num_points=5000)
torus = torus_point_cloud(num_points=5000)

plot_point_cloud(sphere, "Sphere")
plot_point_cloud(torus, "Torus")
plot_point_cloud(klein, "Klein Bottle")
plot_point_cloud(crosscap, "Crosscap")


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

visualize_graph(klein, G_klein, "Klein Bottle with k-NN Graph")
visualize_graph(sphere, G_sphere, "Sphere with k-NN Graph")
visualize_graph(torus, G_torus, "Torus with k-NN Graph")
visualize_graph(crosscap, G_crosscap, "Crosscap with k-NN Graph")
