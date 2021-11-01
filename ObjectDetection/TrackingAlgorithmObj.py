
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as scs
from statstests import StatCompute
from matplotlib.animation import FuncAnimation

class BodyObject(object):
    def __init__(self):
        self.Position_Estimate = None
        self.Velocity_Estimate = None
        self.MeasurementCount = 0
        self.NewMeasurements = []
        self.GuessedDangerClass = None
        self.TrackedFlag = 0



def update_track(CurrBodies , PrevBodies, PotentialBodies, covar):

    covariance = covar
    #compare to potential tracked bodies
    CnfdLvl = 0.7
    alph = (1- CnfdLvl)/2

    for k,p in enumerate(PotentialBodies):
        if p.MeasurementCount < 0:
            del PotentialBodies[k]


    for k,p in enumerate(PotentialBodies):
        point = p.Position_Estimate
        mu =  np.float64(point)
        
        covar = 1.0*covariance @ covariance
        sc = StatCompute(mu,covar)
        changeMade =  0
        for q,n in enumerate(CurrBodies):
            
            out = sc.computePseudoPValue(n.Position_Estimate)
            if( alph > out):
                changeMade += 1
                p.Position_Estimate = n.Position_Estimate
                p.Velocity_Estimate = n.Velocity_Estimate
                p.MeasurementCount = p.MeasurementCount + 1
                del CurrBodies[q]
                
            else:
                pass
        if changeMade == 0:
            p.MeasurementCount = p.MeasurementCount - 1

    dPot = [n.MeasurementCount for n in PotentialBodies]
    #print(dPot)
    #remove duplicate potential tracked bodies
    alph = (1- CnfdLvl)/2
    for k,p in enumerate(PotentialBodies):
        point = p.Position_Estimate
        mu =  np.float64(point)
        covar = 1.0*covariance @ covariance
        sc = StatCompute(mu,covar)
        for q,n in enumerate(PotentialBodies):
            out = sc.computePseudoPValue(n.Position_Estimate)
            if( alph > out) and (k != q):
                
                del PotentialBodies[q]
            else:
                pass            
    

    #Add New To Tracked Mechanism
    #compare association between new and prev
    CnfdLvl = 0.7
    alph = (1- CnfdLvl)/2
    for k,p in enumerate(PrevBodies):
        point = p.Position_Estimate
        mu =  np.float64(point)
        covar = 1.0*covariance @ covariance
        sc = StatCompute(mu,covar)
        for q,n in enumerate(CurrBodies):
            out = sc.computePseudoPValue(n.Position_Estimate)
            
            if( alph > out):
                n.MeasurementCount = p.MeasurementCount + 1
            else:
                pass
               # n.MeasurementCount = p.MeasurementCount 

    

    d = [n.MeasurementCount for n in CurrBodies]

    for i,score in enumerate(d):
        if score > 5:
            PotentialBodies.append(CurrBodies[i])
    return CurrBodies , PrevBodies, PotentialBodies



# update_plot(1)
# print("second round")
# update_plot(1)

# point1 = 1.0*np.array([70, 120])
# covariance = np.array([[100., 0],[0, 100.]])
# rndGen =  scs.multivariate_normal(mean=[0,0], cov=covariance)
# TestP1 = point1 +rndGen.rvs(1)
# TestP2 = point1 +rndGen.rvs(1)

# print(point1)
# print(TestP1)
# print(TestP2)

# mu =  np.float64(TestP1)
# covar = 3*covariance @ covariance
# sc = StatCompute(mu,covar)
# out = sc.computePseudoPValue(TestP2)
# print(out)
# print((1.-0.75)/2)
# fig1, ax1 = plt.subplots()



# rv = scs.multivariate_normal(mean=point1, cov=covar)



# x = np.linspace(0,480,480)
# y = np.linspace(0,720,720)
# X,Y = np.meshgrid(x,y )

# pos = np.dstack((X, Y))
# Z = rv.pdf(pos)
# print(rv.mean)
# #Z = [ rv.pdf([xp, yp])   for yp in y for xp in x]


# ax1.imshow(Z)



# ax1.scatter(point1[0],point1[1])
# ax1.scatter(TestP1[0],TestP1[1])
# ax1.scatter(TestP2[0],TestP2[1])

# ax.scatter(MainPoints[:,0], MainPoints[:,1])
# ax.scatter(CurrPoints[:,0], CurrPoints[:,1])

#ax1.set_xlim([point1[0]-50,point1[0]+50])
#ax1.set_ylim([point1[1]-50,point1[1]+50])
#plt.show()


