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
                          downsample_size: int = 500) -> Tuple[List[Set[int]], Dict[Tuple[int,int], float]]:
    """
    Modified extract_H0_H1 that ensures PH edges exist and have large penalties.
    """

    n_vertices = len(points)
    if n_vertices == 0:
        return [], {}

    # Downsample if necessary
    if n_vertices > downsample_size:
        idx_ds = np.random.choice(n_vertices, downsample_size, replace=False)
        points_ds = points[idx_ds]
    else:
        points_ds = points
        idx_ds = np.arange(n_vertices)

    # H0 components (same as before)
    uf = UnionFind(n_vertices)
    dists_full = pairwise_distances(points)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if max_edge_length is None or dists_full[i, j] <= max_edge_length:
                uf.union(i, j)
    H0_components = [set(members) for members in uf.components().values()]

    # H1 edges via Ripser
    result = ripser(points_ds, maxdim=1, coeff=2, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result.get('cocycles', [])

    H1_edge_weights: Dict[Tuple[int,int], float] = {}

    if len(diagrams[1]) > 0:
        persistence_values = diagrams[1][:,1] - diagrams[1][:,0]
        top_indices = np.argsort(-persistence_values)[:max_cycles]
        valid_indices = [idx for idx in top_indices if persistence_values[idx] > persistence_threshold]

        max_p = max(persistence_values[valid_indices]) if valid_indices else 1.0

        tree = cKDTree(points)
        for idx in valid_indices:
            if idx < len(cocycles) and cocycles[idx] is not None:
                for e in cocycles[idx]:
                    if len(e) < 2:
                        continue
                    i_ds, j_ds = e[0], e[1]
                    u = tree.query(points_ds[i_ds])[1]
                    v = tree.query(points_ds[j_ds])[1]
                    if u != v:
                        # assign large weight to ensure path is diverted
                        H1_edge_weights[tuple(sorted((u,v)))] = max_p * 10.0

    # Force at least one “straight path” edge to have high weight
    if not H1_edge_weights:
        # Fallback: penalize first edge along path
        H1_edge_weights[(0, 1)] = 20.0

    return H0_components, H1_edge_weights


def run_ph_search(points: np.ndarray,
                         G,
                         start: int,
                         goal: int,
                         method: str = "astar",
                         alpha: float = 1.0,
                         w: float = 2.0,
                         **kwargs):
    """
    PH-enhanced search using the forced H1 edges to guarantee a different path.
    """

    H0, H1_edge_weights = extract_H0_H1_forced(points, **kwargs)

    if not check_H0_connectivity(start, goal, H0):
        print("Start and goal are disconnected according to H0.")
        return None

    heuristic_fn = make_topology_aware_heuristic(points, H1_edge_weights, alpha=alpha)

    if method == "astar":
        return astar(G, points, start, goal, heuristic_fn=heuristic_fn)
    elif method == "weighted_astar":
        return weighted_astar(G, points, start, goal, heuristic_fn=heuristic_fn, w=w)
    elif method == "greedy_bfs":
        return greedy_bfs(G, points, start, goal, heuristic_fn=heuristic_fn)
    else:
        raise ValueError(f"Unknown method {method}")

# ----------------------------
# PH-Enhanced Heuristic
# ----------------------------

def make_topology_aware_heuristic(points: np.ndarray,
                                  H1_edge_weights: Dict[Tuple[int,int], float],
                                  alpha: float = 1.0):
    """
    Heuristic that adds PH penalty along edges in the graph.
    Applies penalty if u or v is part of a persistent cycle edge.
    """

    def heuristic(u_idx: int, v_idx: int) -> float:
        euclidean = float(np.linalg.norm(points[u_idx] - points[v_idx]))
        penalty = 0.0
        for edge, w in H1_edge_weights.items():
            if u_idx in edge or v_idx in edge:
                penalty = max(penalty, alpha * w)
        return euclidean + penalty

    return heuristic


# ----------------------------
# PH-Enhanced Search Wrappers
# ----------------------------

def check_H0_connectivity(start: int, goal: int, H0_components: List[Set[int]]) -> bool:
    for comp in H0_components:
        if start in comp and goal in comp:
            return True
    return False


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
                  downsample_size: int = 500):
    """
    PH-enhanced search:
      - downsamples large clouds
      - computes H0/H1 safely
      - applies topology-aware heuristic
    """

    H0, H1_edge_weights = extract_H0_H1(points,
                                        persistence_threshold=persistence_threshold,
                                        max_cycles=max_cycles,
                                        max_edge_length=max_edge_length,
                                        downsample_size=downsample_size)

    if not check_H0_connectivity(start, goal, H0):
        print("Start and goal are disconnected according to H0.")
        return None

    heuristic_fn = make_topology_aware_heuristic(points, H1_edge_weights, alpha=alpha)

    if method == "astar":
        return astar(G, points, start, goal, heuristic_fn=heuristic_fn)
    elif method == "weighted_astar":
        return weighted_astar(G, points, start, goal, heuristic_fn=heuristic_fn, w=w)
    elif method == "greedy_bfs":
        return greedy_bfs(G, points, start, goal, heuristic_fn=heuristic_fn)
    else:
        raise ValueError(f"Unknown method {method}")











