import numpy as np
def sdf_sdbox(x,b):
    x=np.array(x)
    x=np.abs(x)
    d=np.zeros((x.shape[0],x.shape[1]))
    d[:,0]=x[:,0]-b[0]
    d[:,1]=x[:,1]-b[1]
    tmp=np.maximum(d,0.0)
    return np.sqrt(tmp[:,0]**2+tmp[:,1]**2)+np.minimum(np.maximum(d[:,0],d[:,1]),0.0)