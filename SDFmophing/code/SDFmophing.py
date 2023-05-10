import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


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

def sdf_sdbox(x,b):
    x=np.array(x)
    x=np.abs(x)
    d=np.zeros((x.shape[0],x.shape[1]))
    d[:,0]=x[:,0]-b[0]
    d[:,1]=x[:,1]-b[1]
    tmp=np.maximum(d,0.0)
    return np.sqrt(tmp[:,0]**2+tmp[:,1]**2)+np.minimum(np.maximum(d[:,0],d[:,1]),0.0)

# https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.html
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

def sdOctogon(x, r=0.5):
     x = np.array(x)
     k = np.array([-0.9238795325, 0.3826834323, 0.4142135623])
     kxy = np.array([k[0], k[1]])
     kyx = np.array([k[1], k[0]])
     x = np.abs(x)
     x -= 2.0 * np.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
     x -= 2.0 * np.minimum(x.dot(kyx), 0.0)[:,None] * kyx[None,:]
     x[:,0] -= np.clip(x[:,0],r*k[2],r*k[3])
     x[:,1] -= r
     length_x = np.sqrt(np.sum(x**2, 1))
     return np.array(length_x*np.sign(x[:,1]))

def sdfHexagon(x, r=0.5):
     x = np.array(x)
     k = np.array([-0.866025404,0.5,0.577350269])
     kxy = np.array([k[0], k[1]])
     x = np.abs(x)
     x -= 2.0 * np.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
     x[:,0] -= np.clip(x[:,0],-r*k[2],r*k[2])
     x[:,1] -= r
     length_x = np.sqrt(np.sum(x**2, 1))
     return np.array(length_x*np.sign(x[:,1]))


def sdf_circle(x, r=0.5):
     return np.sqrt(np.sum(x**2, axis=1)) - r

def sample_grid(resolution, low=-1.0, high=1.0):
     idx = np.linspace(low, high, num=resolution)
     x, y = np.meshgrid(idx, idx)
     V = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)

     return np.array(V)

grid_size = 32

x = sample_grid(grid_size)
y = [0.5,0.5]

f1 = sdf_hexagram(x)
f2 = sdf_circle(x)
f3= sdfHexagon(x)
f4 = sdf_sdbox(x,y)

# Sample points inside f1
idx1 = f1 < 0
x1_in = x[idx1, :]

# Color-map
sdf_cm = mpl.colors.LinearSegmentedColormap.from_list('SDF',
[(0,'#eff3ff'),(0.5,'#3182bd'),(0.5,'#31a354'),(1,'#e5f5e0')], N=256)

plt.figure()
plt.scatter(x1_in[:,0], x1_in[:,1], c=f1[idx1], cmap=sdf_cm)
plt.axis('equal')
plt.axis("off")
plt.show()

idx2 = f2 < 0
x2_in = x[idx2, :]

plt.figure()
plt.scatter(x2_in[:,0], x2_in[:,1], c=f2[idx2], cmap=sdf_cm)
plt.axis('equal')
plt.axis("off")
plt.show()

idx3 = f3 < 0
x3_in = x[idx3, :]

plt.figure()
plt.scatter(x3_in[:,0], x3_in[:,1], c=f3[idx3], cmap=sdf_cm)
plt.axis('equal')
plt.axis("off")
plt.show()

idx4 = f4 < 0
x4_in = x[idx4, :]

plt.figure()
plt.scatter(x4_in[:,0], x4_in[:,1], c=f4[idx4], cmap=sdf_cm)
plt.axis('equal')
plt.axis("off")
plt.show()
plt.show()