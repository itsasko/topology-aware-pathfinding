import numpy as np
from ripser import ripser
from typing import Tuple, Set, Dict, List, Optional
from heuristic_search import astar, weighted_astar, greedy_bfs
from scipy.spatial import cKDTree

# ----------------------------
# Utilities
# ----------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x: int) -> int:
        p = x
        while self.parent[p] != p:
            p = self.parent[p]
        root = p
        p = x
        while self.parent[p] != root:
            nxt = self.parent[p]
            self.parent[p] = root
            p = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> Dict[int, List[int]]:
        roots = [self.find(i) for i in range(len(self.parent))]
        comp: Dict[int, List[int]] = {}
        for i, r in enumerate(roots):
            comp.setdefault(r, []).append(i)
        return comp


# ----------------------------
# PH Preprocessing Functions
# ----------------------------

def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def extract_H0_H1(points: np.ndarray,
                  persistence_threshold: float = 1e-3,
                  max_cycles: int = 3,
                  max_edge_length: Optional[float] = None,
                  downsample_size: int = 500) -> Tuple[List[Set[int]], Dict[int, float]]:
    """
    Extract H0 components and H1 node weights using Ripser.
    Automatically downsamples large point clouds for efficiency.
    """

    n_vertices = len(points)
    if n_vertices == 0:
        return [], {}

    # --- Downsample for PH ---
    if n_vertices > downsample_size:
        idx_ds = np.random.choice(n_vertices, downsample_size, replace=False)
        points_ds = points[idx_ds]
    else:
        points_ds = points
        idx_ds = np.arange(n_vertices)

    # --- Determine max_edge_length if needed ---
    if max_edge_length is None:
        dists = pairwise_distances(points_ds)
        iu = np.triu_indices(len(points_ds), k=1)
        nonzero = dists[iu]
        max_edge_length = float(np.percentile(nonzero, 90)) if nonzero.size > 0 else np.finfo(float).eps

    # --- H0 components via threshold ---
    uf = UnionFind(n_vertices)
    dists_full = pairwise_distances(points)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if dists_full[i, j] <= max_edge_length:
                uf.union(i, j)
    H0_components = [set(members) for members in uf.components().values()]

    # --- H1 via Ripser on downsampled points ---
    result = ripser(points_ds, maxdim=1, coeff=2, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result.get('cocycles', [])

    H1_node_weights: Dict[int, float] = {}

    if len(diagrams[1]) > 0:
        persistence_values = diagrams[1][:, 1] - diagrams[1][:, 0]
        top_indices = np.argsort(-persistence_values)[:max_cycles]

        valid_indices = [idx for idx in top_indices if persistence_values[idx] > persistence_threshold]
        if valid_indices:
            max_p = max(persistence_values[valid_indices])
        else:
            valid_indices = []
            max_p = 0.0

        # Map back to full points using nearest neighbor
        tree = cKDTree(points)
        for idx in valid_indices:
            if idx < len(cocycles) and cocycles[idx] is not None:
                cycle_vertices = np.array(cocycles[idx])
                for v_ds in cycle_vertices:
                    # Find nearest full point
                    nn_idx = tree.query(points_ds[v_ds])[1]
                    H1_node_weights[nn_idx] = max_p
            else:
                # Fallback: assign to all points
                for v in range(n_vertices):
                    H1_node_weights[v] = max_p

    return H0_components, H1_node_weights


# ----------------------------
# PH-Enhanced Search Wrappers
# ----------------------------

def check_H0_connectivity(start: int, goal: int, H0_components: List[Set[int]]) -> bool:
    for comp in H0_components:
        if start in comp and goal in comp:
            return True
    return False


def make_topology_aware_heuristic(points: np.ndarray,
                                  H1_node_weights: Dict[int, float],
                                  alpha: float = 1.0,
                                  weight_fn: Optional[callable] = None):
    if H1_node_weights:
        max_p = max(H1_node_weights.values())
    else:
        max_p = 0.0

    if weight_fn is None:
        if max_p > 0.0:
            def weight_fn_raw(p):
                return p / max_p
            weight_fn = weight_fn_raw
        else:
            weight_fn = lambda p: 0.0

    def heuristic(node_idx: int, goal_idx: int) -> float:
        euclidean = float(np.linalg.norm(points[node_idx] - points[goal_idx]))
        raw_p = H1_node_weights.get(node_idx, 0.0)
        penalty = alpha * float(weight_fn(raw_p))
        return euclidean + penalty

    return heuristic


def run_ph_search(points: np.ndarray,
                  G,
                  start: int,
                  goal: int,
                  method: str = "astar",
                  alpha: float = 1.0,
                  w: float = 2.0,
                  max_edge_length: Optional[float] = None,
                  persistence_threshold: float = 1e-3,
                  max_cycles: int = 3,
                  rips_percentile: float = 90.0,
                  downsample_size: int = 500):
    """
    PH-enhanced search:
      - downsamples large clouds
      - computes H0/H1 safely
      - applies topology-aware heuristic
    """

    H0, H1_node_weights = extract_H0_H1(points,
                                        persistence_threshold=persistence_threshold,
                                        max_cycles=max_cycles,
                                        max_edge_length=max_edge_length,
                                        downsample_size=downsample_size)

    if not check_H0_connectivity(start, goal, H0):
        print("Start and goal are disconnected according to H0.")
        return None

    heuristic_fn = make_topology_aware_heuristic(points, H1_node_weights, alpha=alpha)

    if method == "astar":
        return astar(G, points, start, goal, heuristic_fn=heuristic_fn)
    elif method == "weighted_astar":
        return weighted_astar(G, points, start, goal, heuristic_fn=heuristic_fn, w=w)
    elif method == "greedy_bfs":
        return greedy_bfs(G, points, start, goal, heuristic_fn=heuristic_fn)
    else:
        raise ValueError(f"Unknown method {method}")









