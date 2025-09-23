import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from surfaces import (klein_bottle_point_cloud, crosscap_point_cloud, torus_point_cloud,
                      sphere_point_cloud)


def plot_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="lightcoral")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


sphere = sphere_point_cloud(num_points=5000)
klein = klein_bottle_point_cloud(num_points=5000)
crosscap = crosscap_point_cloud(num_points=10000)
torus = torus_point_cloud(num_points=5000)

plot_point_cloud(sphere, "Sphere")
plot_point_cloud(torus, "Torus")
plot_point_cloud(klein, "Klein Bottle")
plot_point_cloud(crosscap, "Crosscap")
