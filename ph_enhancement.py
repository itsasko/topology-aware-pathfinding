import numpy as np
import gudhi as gd
from typing import Tuple, Set, Dict, List, Iterable, Optional
from heuristic_search import astar, weighted_astar, greedy_bfs

# ----------------------------
# Utilities
# ----------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x: int) -> int:
        # path compression
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
        # union by rank
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
    """
    Efficient pairwise distances matrix (NxN).
    """
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def compute_rips_complex(points: np.ndarray,
                         max_edge_length: Optional[float] = None,
                         max_dim: int = 1,
                         percentile: float = 90.0) -> Tuple[gd.SimplexTree, float]:
    """
    Build Rips complex and return simplex tree plus used max_edge_length.
    If max_edge_length is None, use the given percentile of pairwise distances
    (default 90th) to avoid outlier-driven scales.
    Returns (simplex_tree, used_max_edge_length).
    """
    n = len(points)
    if n == 0:
        raise ValueError("Empty point cloud")

    dists = pairwise_distances(points)
    if max_edge_length is None:
        # use upper-triangle distances only (exclude zeros on diagonal)
        iu = np.triu_indices(n, k=1)
        nonzero = dists[iu]
        if nonzero.size == 0:
            used = 0.0
        else:
            used = float(np.percentile(nonzero, percentile))
    else:
        used = float(max_edge_length)

    # if used == 0 (all points identical or single point), keep a tiny epsilon
    if used == 0.0:
        used = np.finfo(float).eps

    rips = gd.RipsComplex(points=points, max_edge_length=used)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.persistence()  # compute persistence for later queries
    return st, used


def extract_H0_H1(st: gd.SimplexTree,
                  persistence_threshold: float = 1e-3,
                  max_cycles: int = 3) -> Tuple[List[Set[int]], Dict[int, float]]:
    """
    Extract H0 components and H1 node weights using actual persistence generators.
    """
    n_vertices = len([v for v in st.get_skeleton(0)])
    if n_vertices == 0:
        return [], {}

    # Build H0 via union-find
    uf = UnionFind(n_vertices)
    for simplex, _ in st.get_skeleton(1):
        if len(simplex) == 2:
            uf.union(simplex[0], simplex[1])
    H0_components = [set(members) for members in uf.components().values()]

    # Enable persistence generators
    st.flag_persistence_generators()
    intervals = st.persistence_intervals_in_dimension(1)
    if len(intervals) == 0:
        return H0_components, {}

    # Top persistent 1-cycles
    top_intervals = sorted(intervals, key=lambda iv: iv[1] - iv[0], reverse=True)[:max_cycles]

    H1_node_weights: Dict[int, float] = {}
    for gen_simplex, dim, birth, death in st.persistence_generators():
        if dim != 1:
            continue
        for iv in top_intervals:
            # Match generator to interval
            if abs(iv[0]-birth) < 1e-6 and abs(iv[1]-death) < 1e-6:
                for v in gen_simplex:
                    H1_node_weights[v] = death - birth  # persistence as weight

    return H0_components, H1_node_weights



# ----------------------------
# PH-Enhanced Search Wrappers
# ----------------------------

def check_H0_connectivity(start: int, goal: int, H0_components: Iterable[Set[int]]) -> bool:
    """
    Check if start and goal are in the same connected component.
    """
    for comp in H0_components:
        if start in comp and goal in comp:
            return True
    return False


def make_topology_aware_heuristic(points: np.ndarray,
                                  H1_node_weights: Dict[int, float],
                                  alpha: float = 1.0,
                                  weight_fn: Optional[callable] = None):
    """
    Returns a heuristic function(h(node_idx, goal_idx)) that adds a topology penalty.
    - H1_node_weights: raw persistence values per node (may be empty).
    - alpha: global multiplier for topology penalty.
    - weight_fn: optional function to map raw persistence -> [0,1] penalty factor.
                 default: linear normalization by max persistence (or identity if max==0).
    """

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
                  rips_percentile: float = 90.0):
    """
    Main wrapper:
      - builds Rips complex (auto scale via percentile if max_edge_length is None)
      - extracts H0 components and H1 node weights
      - checks H0 connectivity
      - constructs topology-aware heuristic and runs selected search
    Parameters align with previous API; added rips_percentile to control auto-scale selection.
    """
    st, used_max = compute_rips_complex(points, max_edge_length=max_edge_length,
                                        max_dim=1, percentile=rips_percentile)
    H0, H1_node_weights = extract_H0_H1(st, persistence_threshold=persistence_threshold, max_cycles=max_cycles)

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









