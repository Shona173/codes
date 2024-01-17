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