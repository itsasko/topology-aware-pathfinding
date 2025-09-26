import numpy as np
import gudhi as gd
from heuristic_search import astar, weighted_astar, greedy_bfs

# ----------------------------
# PH Preprocessing Functions
# ----------------------------

def compute_rips_complex(points, max_edge_length=None, max_dim=1):
    """
    Build Rips complex and compute persistent homology.
    If max_edge_length is None, auto-compute from point cloud diameter.
    """
    if max_edge_length is None:
        # Estimate: half of cloud's max pairwise distance
        dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        max_edge_length = np.max(dists) * 0.5
        print(f"[PH] Auto max_edge_length set to {max_edge_length:.3f}")

    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips_complex.create_simplex_tree(max_dimension=max_dim)
    st.persistence()
    return st


def extract_H0_H1(st, persistence_threshold=1e-3, max_cycles=3):
    """
    Extract connected components (H0) and nodes in top persistent 1D cycles (H1)
    """
    # H0 components
    H0_components = [v[0] for v in st.get_skeleton(0)]

    # Flag generators before extracting
    st.flag_persistence_generators()
    H1_cycle_nodes = set()

    intervals = st.persistence_intervals_in_dimension(1)
    if len(intervals) == 0:
        print("[PH] No 1D features found")
        return H0_components, H1_cycle_nodes

    # Sort by persistence
    intervals = sorted(intervals, key=lambda x: x[1] - x[0], reverse=True)

    # Take top persistent cycles
    top_intervals = intervals[:max_cycles]
    generators = st.persistence_generators()  # current Gudhi API

    for birth, death in top_intervals:
        if death - birth < persistence_threshold:
            continue
        for simplex, dim, gen_birth, gen_death in generators:
            if dim == 1 and abs(gen_birth - birth) < 1e-6 and abs(gen_death - death) < 1e-6:
                H1_cycle_nodes.update(simplex)

    print(f"[PH] Found H1 nodes: {len(H1_cycle_nodes)}")
    return H0_components, H1_cycle_nodes


# ----------------------------
# PH-Enhanced Search Wrappers
# ----------------------------

def check_H0_connectivity(start, goal, H0_components):
    # Assume everything connected for now
    return True


def topology_aware_heuristic(node_idx, goal_idx, points, H1_nodes, alpha=1.0):
    """Euclidean distance plus penalty if node belongs to H1 cycle"""
    euclidean = np.linalg.norm(points[node_idx] - points[goal_idx])
    penalty = alpha if node_idx in H1_nodes else 0
    return euclidean + penalty


def run_ph_search(points, G, start, goal, method="astar", alpha=1.0, w=2.0,
                  max_edge_length=None, persistence_threshold=1e-3, max_cycles=3):
    """
    Run a search algorithm on graph G using PH enhancements:
    - H0 pruning
    - Topology-aware heuristic (H1)
    """
    # Compute persistent homology
    st = compute_rips_complex(points, max_edge_length=max_edge_length, max_dim=1)
    H0, H1_nodes = extract_H0_H1(st, persistence_threshold=persistence_threshold, max_cycles=max_cycles)

    if not check_H0_connectivity(start, goal, H0):
        print("Start and goal are disconnected according to H0.")
        return None

    # Heuristic
    def heuristic(s, g):
        return topology_aware_heuristic(s, g, points, H1_nodes, alpha)

    # Run selected algorithm
    if method == "astar":
        return astar(G, points, start, goal, heuristic_fn=heuristic)
    elif method == "weighted_astar":
        return weighted_astar(G, points, start, goal, heuristic_fn=heuristic, w=w)
    elif method == "greedy_bfs":
        return greedy_bfs(G, points, start, goal, heuristic_fn=heuristic)
    else:
        raise ValueError(f"Unknown method {method}")








