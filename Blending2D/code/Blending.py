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

for t in np.linspace(0, 1, num_frames):
    alpha = t
    blended_x = alpha * x1 + (1 - alpha) * x2
    
    blended_color = UniformColors((1 - alpha, 0, alpha, 1))
    
    blended_obj = my_scene.add_shape(blended_x, colors=blended_color)
    
    my_scene.render()

    my_scene.display()

my_scene.clear()

min_points = min(len(x1), len(x2))
x1 = x1[:min_points, :]
x2 = x2[:min_points, :]

alpha = 0.5
blended_x = alpha * x1 + (1 - alpha) * x2

overlap_indices = np.intersect1d(np.where(shape1 < 0)[0], np.where(shape2 < 0)[0])
overlap_indices = overlap_indices[overlap_indices < len(blended_x)]
blended_overlap = blended_x[overlap_indices, :]

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