
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


#region [480 720]

point1 = 1.0*np.array([70, 120])
point2 = 1.0*np.array([300, 500])
point3 = 1.0*np.array([200, 120])
CurrPoints = np.vstack((point1, point2, point3))
MainPoints = CurrPoints.copy()
covariance = np.array([[10., 0],[0, 10.]])

rndGen =  scs.multivariate_normal(mean=[0,0], cov=covariance)

TrackedBodies = []
PotentialBodies = []

x = rndGen.rvs(3)

fig, ax = plt.subplots()

PrevBodies = [] 

def update_plot(i):
    global PrevBodies
    global PotentialBodies
    TempTrack = []
    CurrBodies = []
    #Generate data
    rng = np.random.default_rng()

    ax.clear()
    covariance = np.array([[100., 0],[0, 100.]])
    rndGen =  scs.multivariate_normal(mean=[0,0], cov=covariance)

    NoOfNoise = rng.integers(1, high=6, size=1)
    
    NoisyPoints = rng.integers([0,0], high=[480,720], size=(NoOfNoise[0],2))


    


    point1 = 1.0*np.array([70, 120])
    point2 = 1.0*np.array([300, 500])
    point3 = 1.0*np.array([200, 120])

    CurrPoints = np.vstack((point1, point2, point3))
    CurrPoints = CurrPoints + rndGen.rvs(3)
    #Noisy NewFrame is ready:
    CurrPoints = np.vstack((CurrPoints,NoisyPoints))
    
    for n in CurrPoints:
        temp_bod = BodyObject()
        temp_bod.Position_Estimate = n
        CurrBodies.append(temp_bod)

    
    #compare to potential tracked bodies
    CnfdLvl = 0.7
    alph = (1- CnfdLvl)/2
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
                p.MeasurementCount = p.MeasurementCount + 1
                del CurrBodies[q]
            else:
                pass
        if changeMade == 0:
            p.MeasurementCount = p.MeasurementCount - 1

    dPot = [n.MeasurementCount for n in PotentialBodies]
    print(dPot)
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
            print('Got one')
            PotentialBodies.append(CurrBodies[i])


    
    PrevBodies = CurrBodies.copy()
    ax.scatter(MainPoints[:,0], MainPoints[:,1])
    ax.scatter(CurrPoints[:,0], CurrPoints[:,1])
    for n in PotentialBodies:
        ax.scatter(n.Position_Estimate[0], n.Position_Estimate[1])
    ax.set_xlim([0,480])
    ax.set_ylim([0, 720])

    pass

def init_func():
    ax.clear()

anim = FuncAnimation(fig, update_plot, frames=np.arange(0, 10), init_func=init_func)

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
plt.show()


