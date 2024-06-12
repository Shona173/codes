import numpy as np
import pyvista as pv

# SDF関数を定義（直方体の場合）
def sdf_box(x, y, z, box_size):
    dx = np.maximum(np.abs(x) - box_size[0] / 2, 0)
    dy = np.maximum(np.abs(y) - box_size[1] / 2, 0)
    dz = np.maximum(np.abs(z) - box_size[2] / 2, 0)
    return np.sqrt(dx**2 + dy**2 + dz**2) + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0)

# グリッドのサイズと範囲を定義
grid_size = 100
grid_range = np.linspace(-1.5, 1.5, grid_size)
x, y, z = np.meshgrid(grid_range, grid_range, grid_range)

# SDF値を計算
box_size = [1.0, 0.5, 0.3]  # 直方体のサイズ
sdf_values = sdf_box(x, y, z, box_size)

# PyVistaグリッドを作成
grid = pv.UniformGrid()
grid.dimensions = np.array(sdf_values.shape) + 1
grid.origin = (grid_range[0], grid_range[0], grid_range[0])
grid.spacing = (grid_range[1] - grid_range[0], grid_range[1] - grid_range[0], grid_range[1] - grid_range[0])
grid.cell_data["values"] = sdf_values.flatten(order="F")

# SDF値が0の等値面を抽出
contour = grid.contour([0])

# 等値面をプロット
plotter = pv.Plotter()
plotter.add_mesh(contour, color='lightblue')
plotter.add_axes()
plotter.show()
