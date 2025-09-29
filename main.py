from build_graph import build_knn_graph
from surfaces import (
    klein_bottle_point_cloud, crosscap_point_cloud,
    torus_point_cloud, sphere_point_cloud
)
from heuristic_search import astar, weighted_astar, greedy_bfs
from ph_enhancement import run_ph_search
from visualization import compare_pairwise_paths

import random

# -----------------------------
# Utility: generate random start/goal pairs
# -----------------------------
def generate_start_goal_pairs(num_points, num_pairs=3, seed=None):
    if seed is not None:
        random.seed(seed)
    pairs = []
    for _ in range(num_pairs):
        start = random.randint(0, num_points / 2)
        goal = random.randint(num_points / 2, num_points - 1)
        while goal == start:
            goal = random.randint(0, num_points - 1)
        pairs.append((start, goal))
    return pairs


# -----------------------------
# Generate point clouds
# -----------------------------
sphere = sphere_point_cloud(num_points=5000)
klein = klein_bottle_point_cloud(num_points=5000)
crosscap = crosscap_point_cloud(num_points=5000)
torus = torus_point_cloud(num_points=5000)

# -----------------------------
# Build k-NN graphs
# -----------------------------
G_sphere = build_knn_graph(sphere, k=3)
G_torus = build_knn_graph(torus, k=3)
G_klein = build_knn_graph(klein, k=3)
G_crosscap = build_knn_graph(crosscap, k=3)

# -----------------------------
# Choose random start/goal pairs
# -----------------------------
start_goal_indices = generate_start_goal_pairs(len(sphere), num_pairs=3, seed=42)
# start_goal_indices = [(10, 3900), (100, 4900)]

for start, goal in start_goal_indices:
    print(f"\n=== Running experiments: start={start}, goal={goal} ===")

    # -----------------------------
    # Traditional Searches
    # -----------------------------
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

    # -----------------------------
    # PH-enhanced Searches
    # -----------------------------
    path_sphere_astar = run_ph_search(sphere, G_sphere, start, goal, method="astar", alpha=20.0)
    #path_sphere_wa = run_ph_search(sphere, G_sphere, start, goal, method="weighted_astar", alpha=2.0, w=2.0)
    #path_sphere_gbfs = run_ph_search(sphere, G_sphere, start, goal, method="greedy_bfs", alpha=2.0)

    path_klein_astar = run_ph_search(klein, G_klein, start, goal, method="astar", alpha=20.0)
    #path_klein_wa_ph = run_ph_search(klein, G_klein, start, goal, method="weighted_astar", alpha=2.0, w=2.0)
    #path_klein_gbfs_ph = run_ph_search(klein, G_klein, start, goal, method="greedy_bfs", alpha=2.0)

    path_torus_astar = run_ph_search(torus, G_torus, start, goal, method="astar", alpha=20.0)
    #path_torus_wa_ph = run_ph_search(torus, G_torus, start, goal, method="weighted_astar", alpha=2.0, w=2.0)
    #path_torus_gbfs_ph = run_ph_search(torus, G_torus, start, goal, method="greedy_bfs", alpha=2.0)

    path_crosscap_astar = run_ph_search(crosscap, G_crosscap, start, goal, method="astar", alpha=20.0)
    #path_crosscap_wa_ph = run_ph_search(crosscap, G_crosscap, start, goal, method="weighted_astar", alpha=2.0, w=2.0)
    #path_crosscap_gbfs_ph = run_ph_search(crosscap, G_crosscap, start, goal, method="greedy_bfs", alpha=2.0)

    # -----------------------------
    # Pairwise Comparisons
    # -----------------------------
    print("Visualizing comparisons...")

    # Sphere
    compare_pairwise_paths(sphere, G_sphere, path_a, path_sphere_astar, "A* on Sphere")
    #compare_pairwise_paths(sphere, G_sphere, path_wa, path_sphere_wa, "Weighted A* on Sphere")
    #compare_pairwise_paths(sphere, G_sphere, path_gbfs, path_sphere_gbfs, "Greedy BFS on Sphere")

    # Klein Bottle
    compare_pairwise_paths(klein, G_klein, path_klein_a, path_klein_astar, "A* on Klein Bottle")
    #compare_pairwise_paths(klein, G_klein, path_klein_wa, path_klein_wa_ph, "Weighted A* on Klein Bottle")
    #compare_pairwise_paths(klein, G_klein, path_klein_gbfs, path_klein_gbfs_ph, "Greedy BFS on Klein Bottle")

    # Torus
    compare_pairwise_paths(torus, G_torus, path_torus_a, path_torus_astar, "A* on Torus")
    #compare_pairwise_paths(torus, G_torus, path_torus_wa, path_torus_wa_ph, "Weighted A* on Torus")
    #compare_pairwise_paths(torus, G_torus, path_torus_gbfs, path_torus_gbfs_ph, "Greedy BFS on Torus")

    # Crosscap
    compare_pairwise_paths(crosscap, G_crosscap, path_crosscap_a, path_crosscap_astar, "A* on Crosscap")
    #compare_pairwise_paths(crosscap, G_crosscap, path_crosscap_wa, path_crosscap_wa_ph, "Weighted A* on Crosscap")
    #compare_pairwise_paths(crosscap, G_crosscap, path_crosscap_gbfs, path_crosscap_gbfs_ph, "Greedy BFS on Crosscap")








