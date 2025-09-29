import random
import numpy as np

from surfaces import sphere_point_cloud, torus_point_cloud, klein_bottle_point_cloud, crosscap_point_cloud
from build_graph import build_knn_graph
from comparisons import compare_surface_algorithms, experiment_data, print_summary_table

# -----------------------------
# Generate random start/goal pairs
# -----------------------------
def generate_start_goal_pairs(num_points, num_pairs=3, seed=None):
    if seed is not None:
        random.seed(seed)
    pairs = []
    for _ in range(num_pairs):
        start = random.randint(0, num_points // 2)
        goal = random.randint(num_points // 2, num_points - 1)
        while goal == start:
            goal = random.randint(0, num_points - 1)
        pairs.append((start, goal))
    return pairs

# -----------------------------
# Experiment settings
# -----------------------------
point_cloud_sizes = [300, 500, 1000, 3000, 5000]
num_pairs_per_surface = 3
knn_k = 10  # number of neighbors for graph
alpha = 1.0
persistence_threshold = 1e-3

# -----------------------------
# Surfaces
# -----------------------------
surfaces = {
    "Sphere": sphere_point_cloud,
    "Torus": torus_point_cloud,
    "Klein Bottle": klein_bottle_point_cloud,
    "Crosscap": crosscap_point_cloud
}

# -----------------------------
# Run experiments
# -----------------------------
for num_points in point_cloud_sizes:
    print(f"\n=== Running experiments for {num_points} points ===")
    point_clouds = {}
    graphs = {}
    start_goal_pairs_dict = {}

    # Generate points and graphs for all surfaces
    for name, surface_func in surfaces.items():
        print(f"Generating {name} point cloud and graph...")
        points = surface_func(num_points=num_points)
        G = build_knn_graph(points, k=knn_k)
        point_clouds[name] = points
        graphs[name] = G
        start_goal_pairs_dict[name] = generate_start_goal_pairs(num_points, num_pairs=num_pairs_per_surface, seed=42)

    # Run comparisons for each surface
    for name in surfaces.keys():
        compare_surface_algorithms(
            surface_name=name,
            points=point_clouds[name],
            G=graphs[name],
            start_goal_pairs=start_goal_pairs_dict[name],
            num_points=num_points,
            alpha=alpha,
            persistence_threshold=persistence_threshold,
            max_edge_length=None,
            visualize=False  # set True to see divergent path plots
        )

# -----------------------------
# Print summary table and save CSVs
# -----------------------------
print_summary_table(experiment_data)










