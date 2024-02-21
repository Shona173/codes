import numpy as np
def sdf_circle(x, r=0.5):
     return np.sqrt(np.sum(x**2, axis=1)) - r

