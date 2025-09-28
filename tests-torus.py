import unittest
import numpy as np
from ph_enhancement import run_ph_search, extract_H0_H1, check_H0_connectivity


# Torus point cloud generator
def torus_point_cloud(R=2.0, r=1.0, n_major=20, n_minor=15):
    """
    Generates points on a torus surface.
    R = major radius, r = minor radius
    n_major = number of points along the major circle
    n_minor = number along minor circle
    Returns Nx3 array of points
    """
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.vstack([x, y, z]).T


# Fully connected graph with distance-based weights
class FullGraph:
    def __init__(self, points):
        self.points = points
        self.n = len(points)
        self.nodes = list(range(self.n))

    def neighbors(self, u):
        return [v for v in range(self.n) if v != u]

    def __getitem__(self, u):
        return {v: {'weight': np.linalg.norm(self.points[u] - self.points[v])} for v in self.neighbors(u)}


class PHTorusTest(unittest.TestCase):
    def setUp(self):
        self.points = torus_point_cloud(n_major=15, n_minor=10)  # 150 points
        self.G = FullGraph(self.points)
        self.start = 0
        self.goal = 50  # pick two different points

    def test_H0_H1_extraction(self):
        H0, H1_weights = extract_H0_H1(self.points, persistence_threshold=0.05, max_cycles=5, downsample_size=100)
        print(f"H0 components: {len(H0)}, H1 nodes with weights: {len(H1_weights)}")
        # Ensure the torus is connected
        self.assertTrue(len(H0) == 1)
        # At least some nodes should get H1 weights
        self.assertTrue(len(H1_weights) > 0)

    def test_ph_search_astar(self):
        # Ensure start and goal are connected
        H0, _ = extract_H0_H1(self.points, persistence_threshold=0.05, max_cycles=5, downsample_size=100)
        self.assertTrue(check_H0_connectivity(self.start, self.goal, H0))

        path = run_ph_search(self.points, self.G, self.start, self.goal, method="astar", alpha=1.0, downsample_size=100)
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)


if __name__ == "__main__":
    unittest.main()

