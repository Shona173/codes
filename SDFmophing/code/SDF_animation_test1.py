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

import ot #explain Python library using references
import sys 
sys.path.append('../code/')
from tools import circle 
from tools import sdbox 
from tools import hexagram 
from tools import hexagon 
from tools import sample_grid 
from tools import smacof_mds 


grid_size=32

ini=0
fin=0
x = sample_grid.sample_grid(grid_size)
y = [0.5,0.5]

while True:
    ini=input('Choose the initial shape.\n1.circle\n2.box\n3.hexagram\n4.hexagon\n')
    if(ini=='1'):
        initial_shape=circle.sdf_circle(x)
        break
    elif(ini=='2'):
        initial_shape=sdbox.sdf_sdbox(x,y)
        break
    elif(ini=='3'):
        initial_shape=hexagram.sdf_hexagram(x)
        break
    elif(ini=='4'):
        initial_shape=hexagon.sdf_hexagon(x)
        break
    else:
        print('please select the shape')
        continue

while True:
    fin=input('Choose the final shape.\n1.circle\n2.box\n3.hexagram\n4.hexagon\n')
    if(fin==ini):
        print('That is same shape as initial shape. So select other shape.')
        continue
    elif(fin=='1'):
        final_shape=circle.sdf_circle(x)
        break
    elif(fin=='2'):
        final_shape=sdbox.sdf_sdbox(x,y)
        break
    elif(fin=='3'):
        final_shape=hexagram.sdf_hexagram(x)
        break
    elif(fin=='4'):
        final_shape=hexagon.sdf_hexagon(x)
        break
    else:
        print('please select the shape')
        continue

save_dir ='C:\\Users\\taisy\\source\\repos\\Gromov-Wasserstein\\SDFmophing\\images\\'
os.makedirs(save_dir, exist_ok=True)

idx1 = initial_shape < 0
hex_p = x[idx1, :] #make point cloud in initial shape

idx2 = final_shape < 0 #make point cloud in final shape
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
dir="C:\\Users\\taisy\\source\\repos\\ShapeMorphingOT\\SDFmorphing\\images\\" #image directory

ax1 = plt.subplot2grid((1, 1), (0, 0))
ax1.set_aspect('equal')
ax1.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax1.scatter(npos[0][:, 0], npos[0][:, 1], color='r')
filename1=dir+"ax1.png"
plt.savefig(filename1)

ax2 = plt.subplot2grid((1, 1), (0, 0))
ax2.set_aspect('equal')
ax2.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax2.scatter(npost01[1][:, 0], npost01[1][:, 1], color='b')
filename2=dir+"ax2.png"
plt.savefig(filename2)

ax3 = plt.subplot2grid((1, 1), (0, 0))
ax3.set_aspect('equal')
ax3.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax3.scatter(npost01[0][:, 0], npost01[0][:, 1], color='b')
filename3=dir+"ax3.png"
plt.savefig(filename3)

ax4 = plt.subplot2grid((1, 1), (0, 0))
ax4.set_aspect('equal')
ax4.axis("off")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
ax4.scatter(npos[1][:, 0], npos[1][:, 1], color='r')
filename4=dir+"ax4.png"
plt.savefig(filename4)

plt.show()

pick=glob.glob(dir+"\*.png")
fig=plt.figure()
ani=[]
for i in range(len(pick)):
    tmp=Image.open(pick[i])
    ani.append([plt.imshow(tmp)])
anima=animation.ArtistAnimation(fig,ani,interval=300,repeat_delay=1000)
anima.save("SDF_test1.gif")
