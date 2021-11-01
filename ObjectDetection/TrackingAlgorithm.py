
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as scs
from statstests import StatCompute


class BodyObject(object):
    def __init__(self):
        self.Position_Estimate = None
        self.Velocity_Estimate = None
        self.MeasurementCount = 0
        self.NewMeasurements = []
        self.GuessedDangerClass = None
        self.TrackedFlag = 0


#region [480 720]

point1 = 1.0*np.array([70, 120])
point2 = 1.0*np.array([300, 500])
point3 = 1.0*np.array([200, 120])
CurrPoints = np.vstack((point1, point2, point3))
MainPoints = CurrPoints.copy()
covariance = np.array([[10, 0],[0, 10]])

rndGen =  scs.multivariate_normal(mean=[0,0], cov=covariance)

x = rndGen.rvs(3)
for i in range(0, 25):
    CurrPoints = CurrPoints + rndGen.rvs(3)

    if len(PrevPoints):
        pass

    PrevPoints = CurrPoints.copy()

    pass






fig, ax = plt.subplots()
ax.scatter(MainPoints[:,0], MainPoints[:,1])
ax.scatter(CurrPoints[:,0], CurrPoints[:,1])

ax.set_xlim([0,480])
ax.set_ylim([0, 720])
plt.show()
