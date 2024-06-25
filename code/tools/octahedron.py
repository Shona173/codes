import numpy as np
import pyvista as pv

# SDF関数を定義（正八面体の場合）
def sdf_octahedron(x, y, z, size):
    k = np.sqrt(2)
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)
    sdf = (x_abs + y_abs + z_abs - size) / k
    return sdf

# グリッドのサイズと範囲を定義
grid_size = 100
grid_range = np.linspace(-1.5, 1.5, grid_size)
x, y, z = np.meshgrid(grid_range, grid_range, grid_range)

# SDF値を計算
size = 1.0  # 正八面体のサイズ
sdf_values = sdf_octahedron(x, y, z, size)

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
