import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3Dグリッドのサイズ
grid_size = 100
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
z = np.linspace(-1, 1, grid_size)
x, y, z = np.meshgrid(x, y, z)

# 球体の中心と半径
center = np.array([0, 0, 0])
radius = 0.5

# SDFを計算
sdf = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) - radius

# SDFを使ってボリュームを可視化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 等高線をプロット
ax.contour3D(x, y, z, sdf, levels=[0], colors='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

