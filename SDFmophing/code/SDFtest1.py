import numpy as np
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn import manifold
from sklearn.decomposition import PCA

import ot


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

idx1 = hex1 < 0
hex_p = x[idx1, :] #make point cloud in hex

idx2 = cir1 < 0 #make point cloud in ciecle
cir_p = x[idx2, :]

S=2
xs=[[] for j in range(S)]
xs[0]=hex_p
xs[1]=cir_p

n_samples=500

print('culculate pairwise distance')#Compute the pairwise distances for each subset of (N) 2D points.
ns = [len(xs[s]) for s in range(S)]

Cs = [sp.spatial.distance.cdist(xs[s], xs[s]) for s in range(S)]
Cs = [cs / cs.max() for cs in Cs]

ps = [ot.unif(ns[s]) for s in range(S)]
p = ot.unif(n_samples)


lambdast = [[float(i) / 3, float(3 - i) / 3] for i in [1, 2]]

print('calculate GW barycenter')
Ct01 = [0 for i in range(2)]
for i in range(2):
    Ct01[i] = ot.gromov.gromov_barycenters(n_samples, [Cs[0], Cs[1]],
                                           [ps[0], ps[1]
                                            ], p, lambdast[i], 'square_loss',  # 5e-4,
                                           max_iter=100, tol=1e-3)
    clf = PCA(n_components=2)
npos = [smacof_mds(Cs[s], 2) for s in range(S)]

print('MDS')
npost01 = [0, 0]
npost01 = [smacof_mds(Ct01[s], 2) for s in range(2)]
npost01 = [clf.fit_transform(npost01[s]) for s in range(2)]

print('Plotting')
fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot2grid((1, 4), (0, 0))
ax1.set_aspect('equal')
ax1.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax1.scatter(npos[0][:, 0], npos[0][:, 1], color='r')

ax2 = plt.subplot2grid((1, 4), (0, 1))
ax2.set_aspect('equal')
ax2.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax2.scatter(npost01[1][:, 0], npost01[1][:, 1], color='b')

ax3 = plt.subplot2grid((1, 4), (0, 2))
ax3.set_aspect('equal')
ax3.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax3.scatter(npost01[0][:, 0], npost01[0][:, 1], color='b')

ax4 = plt.subplot2grid((1, 4), (0, 3))
ax4.set_aspect('equal')
ax4.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax4.scatter(npos[1][:, 0], npos[1][:, 1], color='r')

plt.show()