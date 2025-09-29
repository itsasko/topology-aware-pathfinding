import unittest
import numpy as np

from build_graph import build_knn_graph
from surfaces import torus_point_cloud
from heuristic_search import weighted_astar
from ph_enhancement import make_topology_aware_heuristic

class TestPHvsTraditional(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        # Smaller, asymmetric torus point cloud
        self.points = torus_point_cloud(num_points=50, R=1.0, r=0.3)

        # Sparse kNN graph to make PH penalty influential
        self.G = build_knn_graph(self.points, k=3)

        self.start, self.goal = 0, len(self.points) // 2

    def test_paths_differ(self):
        # --- Traditional weighted A* ---
        path_trad = weighted_astar(self.G, self.points, self.start, self.goal, w=2.0)

        # --- PH-enhanced heuristic ---
        # Force high penalty along first edge of traditional path
        forced_edge = (path_trad[0], path_trad[1])
        H1_edge_weights = {tuple(sorted(forced_edge)): 50.0}  # huge penalty

        heuristic_fn = make_topology_aware_heuristic(self.points, H1_edge_weights, alpha=1.0)
        path_ph = weighted_astar(self.G, self.points, self.start, self.goal, heuristic_fn=heuristic_fn)

        # Compute path lengths
        def path_length(path, pts):
            return sum(np.linalg.norm(pts[path[i]] - pts[path[i+1]]) for i in range(len(path)-1))

        len_trad = path_length(path_trad, self.points)
        len_ph = path_length(path_ph, self.points)

        print(f"Traditional length: {len_trad:.4f}, PH-enhanced length: {len_ph:.4f}")
        print(f"Traditional path: {path_trad}")
        print(f"PH-enhanced path: {path_ph}")

        # --- ASSERT paths are different ---
        self.assertNotEqual(path_trad, path_ph)
        self.assertNotAlmostEqual(len_trad, len_ph, places=3)

if __name__ == "__main__":
    unittest.main()




