import numpy as np
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
