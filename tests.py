import unittest
import numpy as np
from ph_enhancement import (compute_rips_complex, extract_H0_H1, check_H0_connectivity,
                            make_topology_aware_heuristic, run_ph_search)

# Adapted SimpleGraph for your search implementation
class SimpleGraph:
    def __init__(self, adjacency_matrix):
        self.adj = adjacency_matrix
        self.n = len(adjacency_matrix)
        self.nodes = list(range(self.n))  # required by astar/weighted_astar

    def neighbors(self, u):
        return [v for v, w in enumerate(self.adj[u]) if w > 0]

    def __getitem__(self, u):
        # return dict-style for edge weights
        return {v: {'weight': self.adj[u][v]} for v in self.neighbors(u)}


class PHEnhancementTest(unittest.TestCase):
    def setUp(self):
        # Simple 4-node square
        self.points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        self.adj_matrix = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])
        self.G = SimpleGraph(self.adj_matrix)
        self.start, self.goal = 0, 2

    def test_rips_complex(self):
        st, max_edge = compute_rips_complex(self.points)
        self.assertTrue(st.num_simplices() > 0)
        self.assertTrue(max_edge > 0)

    def test_extract_H0_H1(self):
        st, _ = compute_rips_complex(self.points)
        H0, H1_weights = extract_H0_H1(st)
        self.assertTrue(len(H0) > 0)
        self.assertIsInstance(H1_weights, dict)

    def test_H0_connectivity(self):
        st, _ = compute_rips_complex(self.points)
        H0, _ = extract_H0_H1(st)
        self.assertTrue(check_H0_connectivity(self.start, self.goal, H0))

    def test_topology_aware_heuristic(self):
        st, _ = compute_rips_complex(self.points)
        _, H1_weights = extract_H0_H1(st)
        heuristic = make_topology_aware_heuristic(self.points, H1_weights, alpha=1.0)
        val = heuristic(self.start, self.goal)
        self.assertGreater(val, 0)

    def test_run_ph_search_astar(self):
        path = run_ph_search(self.points, self.G, self.start, self.goal, method="astar")
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)

    def test_run_ph_search_weighted_astar(self):
        path = run_ph_search(self.points, self.G, self.start, self.goal, method="weighted_astar", w=2.0)
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)

    def test_run_ph_search_greedy(self):
        path = run_ph_search(self.points, self.G, self.start, self.goal, method="greedy_bfs")
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)


if __name__ == "__main__":
    unittest.main()

