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
    """
    Klein bottle point cloud using Paul Bourke's parametrization:
    http://paulbourke.net/geometry/klein/
    """
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)

    r = 4 * (1 - np.cos(u) / 2)

    x = np.empty_like(u)
    y = np.empty_like(u)
    z = np.empty_like(u)

    half = (0 <= u) & (u < np.pi)

    # for u in [0, π)
    x[half] = 6 * np.cos(u[half]) * (1 + np.sin(u[half])) + r[half] * np.cos(u[half]) * np.cos(v[half])
    y[half] = 16 * np.sin(u[half]) + r[half] * np.sin(u[half]) * np.cos(v[half])

    # for u in [π, 2π)
    x[~half] = 6 * np.cos(u[~half]) * (1 + np.sin(u[~half])) + r[~half] * np.cos(v[~half] + np.pi)
    y[~half] = 16 * np.sin(u[~half])

    # common definition
    z[:] = r * np.sin(v)

    return np.stack((x, y, z), axis=-1)


def crosscap_point_cloud(num_points=5000, aa=1.0):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, np.pi, num_points)  # Adjusted v range to [0, π]

    x = aa ** 2 * np.sin(u) * np.sin(2 * v) / 2
    y = aa ** 2 * np.sin(2 * u) * np.cos(v) ** 2
    z = aa ** 2 * np.cos(2 * u) * np.cos(v) ** 2

    return np.stack((x, y, z), axis=-1)


def torus_point_cloud(R=1, r=0.3, num_points=5000, z_scale=1.0):
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = (r * np.sin(v)) * z_scale
    return np.stack((x, y, z), axis=-1)

