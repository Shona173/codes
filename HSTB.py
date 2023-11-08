import numpy as np
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image,ImageDraw
import glob

from scipy.spatial.distance import cdist

from sklearn import manifold
from sklearn.decomposition import PCA

import ot 
import sys 


def smacof_mds(C, dim, max_iter=3000, eps=1e-9):
    """
    Returns an interpolated point cloud following the dissimilarity matrix C
    using SMACOF multidimensional scaling (MDS) in specific dimensionned
    target space

    Parameters
    ----------
    C : ndarray, shape (ns, ns)
        dissimilarity matrix
    dim : int
          dimension of the targeted space
    max_iter :  int
        Maximum number of iterations of the SMACOF algorithm for a single run
    eps : float
        relative tolerance w.r.t stress to declare converge

    Returns
    -------
    npos : ndarray, shape (R, dim)
           Embedded coordinates of the interpolated point cloud (defined with
           one isometry)
    """

    rng = np.random.RandomState(seed=3)

    mds = manifold.MDS(
        dim,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity='precomputed',
        n_init=1)
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        n_init=1)
    npos = nmds.fit_transform(C, init=pos)

    return npos

# https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
def sdf_hexagram(x, r=0.5):
     x = np.array(x)
     k = np.array([-0.5, 0.86602540378, 0.57735026919, 1.73205080757])
     kxy = np.array([k[0], k[1]])
     kyx = np.array([k[1], k[0]])
     x = np.abs(x)
     x -= 2.0 * np.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
     x -= 2.0 * np.minimum(x.dot(kyx), 0.0)[:,None] * kyx[None,:]
     x[:,0] -= np.clip(x[:,0],r*k[2],r*k[3])
     x[:,1] -= r
     length_x = np.sqrt(np.sum(x**2, 1))
     return np.array(length_x*np.sign(x[:,1]))

def sdf_circle(x, r=0.5):
     return np.sqrt(np.sum(x**2, axis=1)) - r

def sdf_Hexagon(x, r=0.5):
     x = np.array(x)
     k = np.array([-0.866025404,0.5,0.577350269])
     kxy = np.array([k[0], k[1]])
     x = np.abs(x)
     x -= 2.0 * np.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
     x[:,0] -= np.clip(x[:,0],-r*k[2],r*k[2])
     x[:,1] -= r
     length_x = np.sqrt(np.sum(x**2, 1))
     return np.array(length_x*np.sign(x[:,1]))




def sample_grid(resolution, low=-1.0, high=1.0):
     idx = np.linspace(low, high, num=resolution)
     x, y = np.meshgrid(idx, idx)
     V = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
      
     return np.array(V)
 
grid_size=32
 
x = sample_grid(grid_size)

hex1=sdf_hexagram(x)
cir1=sdf_circle(x)

alpha = 0.5  
blended_frame = Image.blend(hex1,cir1, alpha)

blended_frame.show()