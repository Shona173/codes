
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

def Wp_barycenter(distribution,weight,p):
    def objective(x):
        barycenter = np.sum([weights[i] * np.power(wasserstein_distance(x, distributions[i])**p, p) for i in range(len(distributions))])
        return barycenter

    x0=np.random.uniform(size=len(distribution[0]))
    bounds = [(0, 1) for _ in range(len(distributions[0]))]

    result=minimize(objective,x0,bounds=bounds)
    barycenter=result.x

    return barycenter

distributions =[np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])]
weights = [0.5, 0.5] 
p=2
barycenter = Wp_barycenter(distributions, weights, p)
print("Wp barycenter:", barycenter)