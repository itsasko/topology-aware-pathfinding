import numpy as np


def square_with_wall(n_points=500, size=20, wall_rect=(8, 12, 5, 15), corridor_width=2):

    cloud = []
    x1, x2, y1, y2 = wall_rect
    while len(cloud) < n_points:
        x, y = np.random.rand(2) * size

        inside_wall = (x1 <= x <= x2) and (y1 <= y <= y2)
        inside_corridor = ((x1 + x2) / 2 - corridor_width / 2 <= x <= (x1 + x2) / 2 + corridor_width / 2)

        if not (inside_wall and not inside_corridor):
            cloud.append([x, y])
    return np.array(cloud)


def multiple_circular_obstacles(n_points=500, size=20, obstacles=[(5, 5, 2), (15, 15, 3)]):

    cloud = []
    while len(cloud) < n_points:
        x, y = np.random.rand(2) * size
        if all((x - cx) ** 2 + (y - cy) ** 2 > r ** 2 for cx, cy, r in obstacles):
            cloud.append([x, y])
    return np.array(cloud)


def random_open_space(n_points=500, size=20):
    return np.random.rand(n_points, 2) * size
