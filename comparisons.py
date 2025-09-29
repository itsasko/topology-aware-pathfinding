import time
from typing import List, Tuple, Optional
import numpy as np
from collections import defaultdict

from heuristic_search import astar, weighted_astar, greedy_bfs
from ph_enhancement import run_ph_search
from visualization import compare_pairwise_paths

# -----------------------------
# Data collector for experiments
# -----------------------------
experiment_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# -----------------------------
# Compute path length (None-safe)
# -----------------------------
def path_length(path: Optional[List[int]], points: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return float('inf')  # no path found
    return sum(np.linalg.norm(points[path[i]] - points[path[i+1]]) for i in range(len(path)-1))


# -----------------------------
# Run traditional and PH-enhanced searches
# -----------------------------
def run_search_algorithms(points: np.ndarray,
                          G,
                          start: int,
                          goal: int,
                          alpha: float = 1.0,
                          persistence_threshold: float = 1e-3,
                          max_edge_length: Optional[float] = None,
                          method_list: Optional[List[str]] = None) -> dict:
    if method_list is None:
        method_list = ["astar", "weighted_astar", "greedy_bfs"]

    results = {}

    for method in method_list:
        # -----------------------------
        # Traditional
        # -----------------------------
        start_time = time.time()
        path_trad = None
        try:
            if method == "astar":
                path_trad = astar(G, points, start, goal)
            elif method == "weighted_astar":
                path_trad = weighted_astar(G, points, start, goal, w=2.0)
            elif method == "greedy_bfs":
                path_trad = greedy_bfs(G, points, start, goal)
        except Exception as e:
            print(f"Error in {method} traditional search: {e}")
        end_time = time.time()
        if path_trad is None:
            print(f"Warning: {method.upper()} failed to find a path (start={start}, goal={goal})")
        results[f"{method}_trad"] = {
            "length": path_length(path_trad, points),
            "time": end_time - start_time,
            "path": path_trad
        }

        # -----------------------------
        # PH-enhanced
        # -----------------------------
        start_time = time.time()
        path_ph = None
        try:
            path_ph = run_ph_search(
                points, G, start, goal,
                method=method, alpha=alpha,
                persistence_threshold=persistence_threshold,
                max_edge_length=max_edge_length
            )
        except Exception as e:
            print(f"Error in {method} PH-enhanced search: {e}")
        end_time = time.time()
        if path_ph is None:
            print(f"Warning: PH-{method.upper()} failed to find a path (start={start}, goal={goal})")
        results[f"{method}_ph"] = {
            "length": path_length(path_ph, points),
            "time": end_time - start_time,
            "path": path_ph
        }

    return results


# -----------------------------
# Compare metrics & visualize
# -----------------------------
def compare_surface_algorithms(surface_name: str,
                               points: np.ndarray,
                               G,
                               start_goal_pairs: List[Tuple[int,int]],
                               num_points: int,
                               alpha: float = 1.0,
                               persistence_threshold: float = 1e-3,
                               max_edge_length: Optional[float] = None,
                               visualize: bool = True):
    print(f"\n=== Surface: {surface_name}, num_points={num_points} ===")
    for start, goal in start_goal_pairs:
        print(f"\nStart={start}, Goal={goal}")
        metrics = run_search_algorithms(points, G, start, goal,
                                        alpha=alpha,
                                        persistence_threshold=persistence_threshold,
                                        max_edge_length=max_edge_length)
        for method in ["astar", "weighted_astar", "greedy_bfs"]:
            trad = metrics[f"{method}_trad"]
            ph = metrics[f"{method}_ph"]

            delta_length = ph["length"] - trad["length"] if ph["length"] != float('inf') and trad["length"] != float('inf') else float('nan')

            print(f"{method.upper():12} | Trad: {trad['length']:.4f}, PH: {ph['length']:.4f}, "
                  f"Δlength={delta_length:.4f}, Time (Trad/PH): {trad['time']:.4f}/{ph['time']:.4f}s")

            # Store metrics for summary table
            experiment_data[num_points][surface_name][method].append({
                "trad_length": trad['length'],
                "ph_length": ph['length'],
                "delta_length": delta_length,
                "trad_time": trad['time'],
                "ph_time": ph['time']
            })

            # Visualize divergent paths if both exist
            if visualize and trad["path"] and ph["path"] and ph["path"] != trad["path"]:
                compare_pairwise_paths(points, G, trad["path"], ph["path"],
                                       f"{method.upper()} on {surface_name} (PH-enhanced)")


# -----------------------------
# Print aggregated summary table
# -----------------------------
def print_summary_table(experiment_data):
    import matplotlib.pyplot as plt
    import pandas as pd

    print("\n=== PH-enhanced Pathfinding Summary ===")
    for num_points, surfaces in experiment_data.items():
        print(f"\n--- Point Cloud Size: {num_points} ---")
        data_rows = []
        for surface, algos in surfaces.items():
            for algo, metrics_list in algos.items():
                delta_avg = np.mean([m["delta_length"] for m in metrics_list if not np.isnan(m["delta_length"])])
                trad_time_avg = np.mean([m["trad_time"] for m in metrics_list])
                ph_time_avg = np.mean([m["ph_time"] for m in metrics_list])
                print(f"{surface:12} {algo:12} {delta_avg:12.4f} {trad_time_avg:14.4f} {ph_time_avg:12.4f}")
                data_rows.append([num_points, surface, algo, delta_avg, trad_time_avg, ph_time_avg])

        # Save table as CSV for work/paper
        df = pd.DataFrame(data_rows, columns=["NumPoints", "Surface", "Algorithm", "DeltaLength", "TradTime", "PHTime"])
        df.to_csv(f"experiment_summary_{num_points}.csv", index=False)

        # Optional: bar chart for delta lengths
        fig, ax = plt.subplots(figsize=(8, 5))
        for surface, algos in surfaces.items():
            for algo, metrics_list in algos.items():
                delta_avg = np.mean([m["delta_length"] for m in metrics_list if not np.isnan(m["delta_length"])])
                ax.bar(f"{surface}-{algo}", delta_avg)
        ax.set_ylabel("ΔLength (PH - Trad)")
        ax.set_title(f"Average ΔLength for Point Cloud Size {num_points}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()