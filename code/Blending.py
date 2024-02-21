import numpy as np
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import glob

from scipy.spatial.distance import cdist

from sklearn import manifold
from sklearn.decomposition import PCA

import ot 
import sys 
sys.path.append('../code/')
from tools import circle 
from tools import sdbox
from tools import sample_grid 


from blendify import scene
from blendify.colors import UniformColors, VertexColors
from blendify.materials import PrinsipledBSDFMaterial
from blendify.colors import UniformColors


grid_size=32

x = sample_grid.sample_grid(grid_size)
y = [0.5,0.5]

shape1=circle.sdf_circle(x)
shape2=sdbox.sdf_sdbox(x,y)


idx1 = shape1 < 0
x1 = x[idx1, :] 

idx2 = shape2 < 0 
x2 = x[idx2, :]


plt.scatter(x1[:, 0], x1[:, 1], c='red', label='Shape 1')
plt.show()

plt.scatter(x2[:, 0], x2[:, 1], c='blue', label='Shape 2')
plt.show()

my_scene = scene.Scene()

num_frames = 10

def blend_space_time(alpha, t, f1, f2):
    f1t = np.minimum(-t, f1)
    f2t = np.minimum(t - 1, f2)
    return alpha * f1t + (1 - alpha) * f2t

alpha = 0.5
num_frames = 10
results = []
for t in np.linspace(0, 1, num_frames):
    shape1 = circle_sdf(x)
    shape2 = sdbox_sdf(x, y)
    blend = blend_space_time(alpha, t, shape1, shape2)
    results.append(blend.reshape(grid_size, grid_size))

my_scene.clear()

min_points = min(len(x1), len(x2))
x1 = x1[:min_points, :]
x2 = x2[:min_points, :]

plt.scatter(blended_overlap[:, 0], blended_overlap[:, 1], c='purple', label='Overlap')
plt.legend()
plt.show()

c1 = np.random.rand(len(x1))
c2 = np.random.rand(len(x2))

interpolated_attributes = (1 - alpha) * c1 + alpha * c2

my_scene = scene.Scene()

shape1_color = UniformColors((1, 0, 0, 1))
shape2_color = UniformColors((0, 0, 1, 1))

shape1_obj = my_scene.add_shape(x1, colors=shape1_color)
shape2_obj = my_scene.add_shape(x2, colors=shape2_color)

blended_color = UniformColors((1, 0.5, 0, 1)) 
blended_obj = my_scene.add_shape(blended_x, colors=blended_color)

img=my_scene.render()

plt.show(img)