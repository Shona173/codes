import numpy as np
def sample_grid(resolution, low=-1.0, high=1.0):
     idx = np.linspace(low, high, num=resolution)
     x, y = np.meshgrid(idx, idx)
     V = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
      
     return np.array(V)