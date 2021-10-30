
from numpy.core.fromnumeric import std
import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt



class StatCompute(object):
    def __init__(self, mu, covar):
        self.mu = mu
        self.covar = covar
        self.StatFunc = scs.multivariate_normal(mean=self.mu, cov=self.covar)
        self.MaxPVal = None

        testPoint = np.array([1e-7, 1e-7])
        DiffVec = (testPoint - self.mu) / np.linalg.norm((testPoint -self.mu))
        scalar = np.linspace(0., 1., 300)
        Array = [ testPoint + p*6.0* DiffVec   for p in scalar ]
        stat = self.StatFunc.pdf(Array)
        self.MaxPVal = np.trapz(stat)
        print(self.MaxPVal)

    def computePseudoPValue(self, point):
        testPoint = point
        DiffVec = (testPoint - self.mu) / np.linalg.norm((testPoint -self.mu))
        scalar = np.linspace(0., 1., 300)
        Array = [ testPoint + p*6.0* DiffVec   for p in scalar ]
        stat = self.StatFunc.pdf(Array)
        return 1-(np.trapz(stat)/self.MaxPVal)
        
    
mu = [0,0]
covar = 0.9*np.array([[1, 0],[0, 1]])
sc = StatCompute(mu,covar)


point = np.array([0.03,-0.03])
out = sc.computePseudoPValue( point)
print("output ", out)



rv = scs.multivariate_normal(mean=mu, cov=covar)



x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X = np.meshgrid(x )
Y = np.meshgrid(y )




#max


#Find outpoint





Z = [ rv.pdf([xp, yp])   for yp in y for xp in x]


fig, ax = plt.subplots()
Z = np.reshape(Z, (100,100))
ax.imshow(Z)



ax.scatter( len(x)/2+ mu[0]*len(x)/2, len(x)/2+ mu[1]*len(x)/2)
ax.scatter( len(x)/2+ point[0]*len(x)/2, len(x)/2+ point[1]*len(x)/2)
plt.show()




