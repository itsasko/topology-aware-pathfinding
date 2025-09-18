import numpy as np


def sphere_point_cloud(radius=1.0, num_points=5000):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.stack((x, y, z), axis=-1)


def klein_bottle_point_cloud(num_points=5000):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    x = (2 / 15) * (3 + 5 * np.cos(u) * np.sin(u)) * np.sin(v)
    y = (1 / 15) * (3 + 5 * np.cos(u) * np.sin(u)) * np.cos(v)
    z = (2 / 15) * np.sin(u)
    return np.stack((x, y, z), axis=-1)


def crosscap_point_cloud(num_points=5000):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    x = np.sin(2 * u) * np.sin(v) ** 2
    y = np.sin(u) * np.sin(2 * v)
    z = np.cos(u) * np.sin(v)
    return np.stack((x, y, z), axis=-1)


def torus_point_cloud(R=1, r=0.3, num_points=5000):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack((x, y, z), axis=-1)


def two_holed_torus_point_cloud(num_points=5000):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    x = (2 + np.cos(u) + np.cos(2 * u) / 2) * np.cos(v)
    y = (2 + np.cos(u) + np.cos(2 * u) / 2) * np.sin(v)
    z = np.sin(u) + np.sin(2 * u) / 2
    return np.stack((x, y, z), axis=-1)

def trefoil_knot(num_points=5000):
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    return np.stack((x, y, z), axis=-1)

def figure_eight_knot(num_points=5000):
    t = np.linspace(0, 2*np.pi, num_points)
    x = (2 + np.cos(2*t)) * np.cos(3*t)
    y = (2 + np.cos(2*t)) * np.sin(3*t)
    z = np.sin(4*t)
    return np.stack((x, y, z), axis=-1)
