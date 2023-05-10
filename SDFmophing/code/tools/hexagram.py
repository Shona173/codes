import numpy as np
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