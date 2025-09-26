from build_graph import build_knn_graph
from surfaces import (klein_bottle_point_cloud, crosscap_point_cloud, torus_point_cloud,
                      sphere_point_cloud)
from heuristic_search import astar, weighted_astar, greedy_bfs
from ph_enhancement import run_ph_search
from visualization import plot_point_cloud, visualize_graph, visualize_path

# -----------------------------
# Generate point clouds
# -----------------------------
sphere = sphere_point_cloud(num_points=500)
klein = klein_bottle_point_cloud(num_points=500)
crosscap = crosscap_point_cloud(num_points=500)
torus = torus_point_cloud(num_points=500)

# -----------------------------
# Build k-NN graphs (for traditional heuristics)
# -----------------------------
G_sphere = build_knn_graph(sphere, k=8)
G_torus = build_knn_graph(torus, k=8)
G_klein = build_knn_graph(klein, k=8)
G_crosscap = build_knn_graph(crosscap, k=8)

# -----------------------------
# Define start and goal
# -----------------------------
start, goal = 0, 100  # indices in point clouds

# -----------------------------
# Run traditional searches
# -----------------------------
# Sphere
path_a = astar(G_sphere, sphere, start, goal)
path_wa = weighted_astar(G_sphere, sphere, start, goal, w=2.0)
path_gbfs = greedy_bfs(G_sphere, sphere, start, goal)

# Klein Bottle
path_klein_a = astar(G_klein, klein, start, goal)
path_klein_wa = weighted_astar(G_klein, klein, start, goal, w=2.0)
path_klein_gbfs = greedy_bfs(G_klein, klein, start, goal)

# Torus
path_torus_a = astar(G_torus, torus, start, goal)
path_torus_wa = weighted_astar(G_torus, torus, start, goal, w=2.0)
path_torus_gbfs = greedy_bfs(G_torus, torus, start, goal)

# Crosscap
path_crosscap_a = astar(G_crosscap, crosscap, start, goal)
path_crosscap_wa = weighted_astar(G_crosscap, crosscap, start, goal, w=2.0)
path_crosscap_gbfs = greedy_bfs(G_crosscap, crosscap, start, goal)

# -----------------------------
# PH-enhanced searches
# -----------------------------
path_sphere_astar = run_ph_search(sphere, G_sphere, start, goal, method="astar", alpha=0.1)
path_sphere_wa = run_ph_search(sphere, G_sphere, start, goal, method="weighted_astar", alpha=0.1, w=2.0)
path_sphere_gbfs = run_ph_search(sphere, G_sphere, start, goal, method="greedy_bfs", alpha=0.1)

path_klein_astar = run_ph_search(klein, G_klein, start, goal, method="astar", alpha=0.1)
path_klein_wa = run_ph_search(klein, G_klein, start, goal, method="weighted_astar", alpha=0.1, w=2.0)
path_klein_gbfs = run_ph_search(klein, G_klein, start, goal, method="greedy_bfs", alpha=0.1)

path_torus_astar = run_ph_search(torus, G_torus, start, goal, method="astar", alpha=0.1)
path_torus_wa = run_ph_search(torus, G_torus, start, goal, method="weighted_astar", alpha=0.1, w=2.0)
path_torus_gbfs = run_ph_search(torus, G_torus, start, goal, method="greedy_bfs", alpha=0.1)

path_crosscap_astar = run_ph_search(crosscap, G_crosscap, start, goal, method="astar", alpha=0.1)
path_crosscap_wa = run_ph_search(crosscap, G_crosscap, start, goal, method="weighted_astar", alpha=0.1, w=2.0)
path_crosscap_gbfs = run_ph_search(crosscap, G_crosscap, start, goal, method="greedy_bfs", alpha=0.1)

# -----------------------------
# Visualizations (optional)
# -----------------------------
# visualize_path(sphere, G_sphere, path_a, "A* on Sphere")
# visualize_path(sphere, G_sphere, path_sphere_astar, "PH-A* on Sphere")
# visualize_path(sphere, G_sphere, path_wa, "Weighted A* on Sphere")
# visualize_path(sphere, G_sphere, path_sphere_wa, "PH-Weighted A* on Sphere")
# visualize_path(sphere, G_sphere, path_gbfs, "Greedy BFS on Sphere")
# visualize_path(sphere, G_sphere, path_sphere_gbfs, "PH-Greedy BFS on Sphere")
#
# visualize_path(klein, G_klein, path_klein_a, "A* on Klein Bottle")
# visualize_path(klein, G_klein, path_klein_astar, "PH-A* on Klein Bottle")
# visualize_path(klein, G_klein, path_klein_wa, "Weighted A* on Klein Bottle")
# visualize_path(klein, G_klein, path_klein_wa, "PH-Weighted A* on Klein Bottle")
# visualize_path(klein, G_klein, path_klein_gbfs, "Greedy BFS on Klein Bottle")
# visualize_path(klein, G_klein, path_klein_gbfs, "PH-Greedy BFS on Klein Bottle")
#
# visualize_path(torus, G_torus, path_torus_a, "A* on Torus")
# visualize_path(torus, G_torus, path_torus_astar, "PH-A* on Torus")
# visualize_path(torus, G_torus, path_torus_wa, "Weighted A* on Torus")
# visualize_path(torus, G_torus, path_torus_wa, "PH-Weighted A* on Torus")
# visualize_path(torus, G_torus, path_torus_gbfs, "Greedy BFS on Torus")
# visualize_path(torus, G_torus, path_torus_gbfs, "PH-Greedy BFS on Torus")
#
# visualize_path(crosscap, G_crosscap, path_crosscap_a, "A* on Crosscap")
# visualize_path(crosscap, G_crosscap, path_crosscap_astar, "PH-A* on Crosscap")
# visualize_path(crosscap, G_crosscap, path_crosscap_wa, "Weighted A* on Crosscap")
# visualize_path(crosscap, G_crosscap, path_crosscap_wa, "PH-Weighted A* on Crosscap")
# visualize_path(crosscap, G_crosscap, path_crosscap_gbfs, "Greedy BFS on Crosscap")
# visualize_path(crosscap, G_crosscap, path_crosscap_gbfs, "PH-Greedy BFS on Crosscap")

from visualization import compare_pairwise_paths

# Sphere examples
compare_pairwise_paths(sphere, G_sphere, path_a, path_sphere_astar, "A*")
compare_pairwise_paths(sphere, G_sphere, path_wa, path_sphere_wa, "Weighted A*")
compare_pairwise_paths(sphere, G_sphere, path_gbfs, path_sphere_gbfs, "Greedy BFS")

# Klein Bottle
compare_pairwise_paths(klein, G_klein, path_klein_a, path_klein_astar, "A*")
compare_pairwise_paths(klein, G_klein, path_klein_wa, path_klein_wa, "Weighted A*")
compare_pairwise_paths(klein, G_klein, path_klein_gbfs, path_klein_gbfs, "Greedy BFS")









