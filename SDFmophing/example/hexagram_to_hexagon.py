import numpy as np
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn import manifold
from sklearn.decomposition import PCA

import ot
import sys 
sys.path.append('../code/')
from tools import hexagram
from tools import hexagon
from tools import sample_grid
from tools import smacof_mds


grid_size=32

x = sample_grid.sample_grid(grid_size)

hexagram1=hexagram.sdf_hexagram(x)
hexagon1=hexagon.sdf_hexagon(x)

idx1 = hexagram1 < 0
hex_p = x[idx1, :] 

idx2 = hexagon1 < 0 
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
npos = [smacof_mds.smacof_mds(Cs[s], 2) for s in range(S)]

print('MDS')
npost01 = [0, 0]
npost01 = [smacof_mds.smacof_mds(Ct01[s], 2) for s in range(2)]
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
