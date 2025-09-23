import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def build_knn_graph(points, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i])
        for j in indices[i][1:]:
            d = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=d)
    return G