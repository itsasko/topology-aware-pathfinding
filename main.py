import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from surfaces import (klein_bottle_point_cloud, crosscap_point_cloud, torus_point_cloud, two_holed_torus_point_cloud,
                      sphere_point_cloud, trefoil_knot, figure_eight_knot)


def plot_point_cloud(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


sphere = sphere_point_cloud()
klein = klein_bottle_point_cloud()
crosscap = crosscap_point_cloud()
torus = torus_point_cloud()
two_torus = two_holed_torus_point_cloud()

plot_point_cloud(sphere, "Sphere")
plot_point_cloud(klein, "Klein Bottle")
plot_point_cloud(crosscap, "Crosscap")
plot_point_cloud(torus, "Torus")
plot_point_cloud(two_torus, "Two-Holed Torus")


def plot_knot(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], lw=1)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.manager.set_window_title(title)
    plt.show()


trefoil = trefoil_knot()
figure8 = figure_eight_knot()

plot_knot(trefoil, "Trefoil Knot")
plot_knot(figure8, "Figure-Eight Knot")
