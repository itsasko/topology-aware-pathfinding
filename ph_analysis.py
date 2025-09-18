import dionysus as d
import numpy as np


def compute_persistence(point_cloud, max_dim=2, max_radius=3.0):
    """
    Computes persistent homology for a given point cloud.

    Args:
        point_cloud (np.array): array of points shape=(n_points, 2)
        max_dim (int): maximum homology dimension (0,1,2)
        max_radius (float): maximum radius for Vietoris-Rips filtration

    Returns:
        diagrams (list): list of diagrams for H0, H1, H2
    """
    filt = d.fill_rips(point_cloud.tolist(), max_dim, max_radius)
    persistence = d.homology_persistence(filt)
    diagrams = d.init_diagrams(persistence, filt)
    return diagrams


def extract_stable_cycles(diagrams, min_persistence=0.5):
    """
    Extracts stable H1 cycles (obstacles) from persistent homology diagrams.

    Args:
        diagrams (list): output from compute_persistence
        min_persistence (float): minimum "lifespan" of a cycle to consider it a real obstacle

    Returns:
        obstacles (list): list of cycles as (birth, death)
    """
    h1 = diagrams[1]  # H1 = cycles
    obstacles = [(pt.birth, pt.death) for pt in h1 if pt.death - pt.birth > min_persistence]
    return obstacles


def print_diagrams(diagrams):
    """
    Prints H0, H1, H2 diagrams to console.
    """
    for i, dg in enumerate(diagrams):
        print(f"H{i}:")
        for pt in dg:
            birth, death = pt.birth, pt.death
            if death == float('inf'):
                print(f"  born at {birth:.2f}, persists to infinity")
            else:
                print(f"  born at {birth:.2f}, dies at {death:.2f}")
