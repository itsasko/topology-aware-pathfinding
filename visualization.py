import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np

def plot_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="lightcoral")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


def visualize_graph(points, G, title="Graph on Point Cloud"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="lightcoral", alpha=0.7)

    for (i, j) in G.edges():
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c="gray", linewidth=0.5, alpha=0.5)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


def visualize_path(points, G, path, title="Path Visualization", path_color="blue"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="lightcoral", alpha=0.7)

    for (i, j) in G.edges():
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c="gray", linewidth=0.3, alpha=0.3)

    if path:
        for u, v in zip(path[:-1], path[1:]):
            p1, p2 = points[u], points[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c=path_color, linewidth=2)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def compare_pairwise_paths(points, G, standard_path, ph_path, alg_name, title_suffix=""):
    """
    Compare a standard and PH-enhanced path for a single algorithm.

    standard_path: path from standard algorithm
    ph_path: path from PH-enhanced version
    alg_name: string, e.g., "A*", "Weighted A*", "Greedy BFS"
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="lightcoral", alpha=0.5)

    # draw graph edges faintly
    for (i, j) in G.edges():
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c="gray", linewidth=0.2, alpha=0.2)

    # colors for standard and PH-enhanced
    paths = [(standard_path, "blue", alg_name),
             (ph_path, "green", f"PH-{alg_name}")]

    legend_lines = []
    for path, color, label in paths:
        if path:
            for u, v in zip(path[:-1], path[1:]):
                p1, p2 = points[u], points[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        c=color, linewidth=2)
            legend_lines.append(Line2D([0], [0], color=color, lw=2, label=label))

    ax.set_title(f"{alg_name} vs PH-{alg_name}" + (f" - {title_suffix}" if title_suffix else ""))
    ax.set_box_aspect([1, 1, 1])
    ax.legend(handles=legend_lines)
    plt.show()

