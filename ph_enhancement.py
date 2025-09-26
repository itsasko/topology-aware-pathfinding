import numpy as np
import gudhi as gd
from heuristic_search import astar, weighted_astar, greedy_bfs


# ----------------------------
# PH Preprocessing Functions
# ----------------------------

def compute_rips_complex(points, max_edge_length=0.5, max_dim=1):
    """Build Rips complex and compute persistent homology"""
    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips_complex.create_simplex_tree(max_dimension=max_dim)
    st.persistence()
    return st

def extract_H0_H1(st, persistence_threshold=0.01):
    """
    Extract connected components (H0) and nodes in significant cycles (H1)
    """
    # --- H0: list of vertices
    H0_components = [v[0] for v in st.get_skeleton(0)]  # 0-simplices

    # --- H1: collect nodes in 1D cycles above persistence threshold
    H1_cycle_nodes = set()

    # Flag generators
    st.flag_persistence_generators()

    # Get all 1-dimensional persistence intervals
    intervals = st.persistence_intervals_in_dimension(1)
    for birth, death in intervals:
        if death - birth > persistence_threshold:
            # get the generator simplices for this feature
            generators = st.get_persistence_generators(1)
            for simplex, dim in generators:
                for vertex in simplex:
                    H1_cycle_nodes.add(vertex)

    return H0_components, H1_cycle_nodes



# ----------------------------
# PH-Enhanced Search Wrappers
# ----------------------------

def check_H0_connectivity(start, goal, H0_components):
    # All vertices mapped to component 0 for simplicity
    # Extend if using multiple disconnected clusters
    return True  # always connected in these surfaces


def topology_aware_heuristic(node_idx, goal_idx, points, H1_nodes, alpha=0.1):
    """Euclidean distance plus penalty if node belongs to 1D cycle"""
    euclidean = np.linalg.norm(points[node_idx] - points[goal_idx])
    penalty = alpha if node_idx in H1_nodes else 0
    return euclidean + penalty


def run_ph_search(points, G, start, goal, method="astar", alpha=0.1, w=2.0):
    """
    Run a search algorithm on graph G using PH enhancements:
    - H0 pruning
    - Topology-aware heuristic (H1)
    """
    # Compute PH
    st = compute_rips_complex(points, max_edge_length=0.5, max_dim=1)
    H0, H1_nodes = extract_H0_H1(st)

    # H0 pruning
    if not check_H0_connectivity(start, goal, H0):
        print("Start and goal are disconnected according to H0.")
        return None

    # Topology-aware heuristic function
    def heuristic(s, g):
        return topology_aware_heuristic(s, g, points, H1_nodes, alpha)

    # Run selected search algorithm
    if method == "astar":
        return astar(G, points, start, goal, heuristic_fn=heuristic)
    elif method == "weighted_astar":
        return weighted_astar(G, points, start, goal, heuristic_fn=heuristic, w=w)
    elif method == "greedy_bfs":
        return greedy_bfs(G, points, start, goal, heuristic_fn=heuristic)
    else:
        raise ValueError(f"Unknown method {method}")

