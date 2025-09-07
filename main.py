
import matplotlib.pyplot as plt
from point_clouds import square_with_wall, multiple_circular_obstacles

cloud1 = square_with_wall(n_points=500)
plt.scatter(cloud1[:,0], cloud1[:,1], s=10)
plt.title("Square Map with Wall & Corridor")
plt.show()

cloud2 = multiple_circular_obstacles(n_points=500, obstacles=[(5,5,2),(15,15,3)])
plt.scatter(cloud2[:,0], cloud2[:,1], s=10)
plt.title("Multiple Circular Obstacles")
plt.show()
