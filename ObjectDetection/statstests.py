
from numpy.core.fromnumeric import std
import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt



class StatCompute(object):
    def __init__(self, mu, covar, conf_lvl = 0.9):
        self.mu = mu
        self.covar = covar
        self.StatFunc = scs.multivariate_normal(mean=self.mu, cov=self.covar)
        self.MaxPVal = None
        self.alphVal = (1.- conf_lvl)/2.

        testPoint = np.array([mu[0] + 1e-10, mu[1] + 1e-10])
        DiffVec = (testPoint - self.mu) / np.linalg.norm((testPoint - self.mu))
        scalar = np.linspace(0., 1., 350)
        Array = [ testPoint + p*6.0*self.covar[0,0]* DiffVec   for p in scalar ]
        stat = self.StatFunc.pdf(Array)
        self.MaxPVal = np.trapz(stat)
        

    def computePseudoPValue(self, point):
        #print(self.MaxPVal )
        testPoint = point
        DiffVec = (testPoint - self.mu) / np.linalg.norm((testPoint -self.mu))
        scalar = np.linspace(0., 1., 350)
        Array = [ testPoint + p*6.0*(self.covar[0,0])* DiffVec   for p in scalar ]
        stat = self.StatFunc.pdf(Array)
        
        return 1-(np.trapz(stat)/self.MaxPVal)



if __name__ == "__main__":
    mu = [0,0]
    covar = 0.9*np.array([[20, 0],[0, 20]])
    sc = StatCompute(mu,covar)


    point = np.array([0.00,-0.00])
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




