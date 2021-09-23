
from UsefulClasses import DroneObject, SimulatorObject, LocalizerObject

import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
from sklearn.preprocessing import normalize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def main():
    
    fg, ax= plt.subplots(4,1)

    fig = plt.figure()
    axs = plt.axes(projection='3d')
    Invader =    DroneObject()
    Def1 =    DroneObject()
    Def2 =    DroneObject()

    Invader.PositionVector = np.array([1, -1,  1])
    Def1.PositionVector = np.array([2, -3,  1])
    Def2.PositionVector = np.array([2, -3,  3])

    DroneList = [Invader, Def1, Def2]
    

    SimObj = SimulatorObject(DroneList)

    def GenPaths(SimObj, DroneList):

        #Invader Path

        P1 = [8, 8,  1]
        P2 = [2.5,6.2, 2]
        Speed = 0.5

        SimObj.genLinearPath(DroneList[0], P1, P2, Speed)

        #defender 1 Path
        
        P1 = [0, 0,  0.0]
        P2 = [4,2, 2]

        #P2 = [0,1,1]
        Speed = 0.75

        SimObj.genLinearPath(DroneList[1], P1, P2, Speed)


        #defender 2 Path
        
        P1 = [14, 0,  0]
        P2 = [7,2, 2]
        Speed = 0.75

        SimObj.genLinearPath(DroneList[2], P1, P2, Speed)


    Invader.PositionCovar = (1e-2)*np.eye(3)
    GenPaths(SimObj, DroneList)

    SimObj.runSim()

    Obs = Def1.retrieveRayVect(Invader)
    Obs = Obs + Def1.PathLog
    Obs_n = np.vstack((Def1.PathLog[15,:], Obs[15,:]))

    localizer = LocalizerObject()

    localizer.ResolveLocationCam(Def1, Invader)

    X1 =  Def1.PathLog[:,0]
    Y1 =  Def1.PathLog[:,1]
    Z1 =  Def1.PathLog[:,2]

    X2 =  Def2.PathLog[:,0]
    Y2 =  Def2.PathLog[:,1]
    Z2 =  Def2.PathLog[:,2]

    X3 =  Invader.PathLog[:,0]
    Y3 =  Invader.PathLog[:,1]
    Z3 =  Invader.PathLog[:,2]

    X4 =  localizer.EstimatedPath[:,0] 
    Y4 =  localizer.EstimatedPath[:,1] 
    Z4 =  localizer.EstimatedPath[:,2] 


    


    axs.plot3D(X1,Y1,Z1)
    axs.plot3D(X2,Y2,Z2)
    axs.plot3D(X3,Y3,Z3)
    axs.plot3D(X4,Y4,Z4)
    
    #axs.plot3D(Obs_n[:,0],Obs_n[:,1],Obs_n[:,2])

    T = SimObj.TimeVect

    ax[0].plot(T,X4)
    ax[0].plot(T,X3)

    ax[1].plot(T,Y4)
    ax[1].plot(T,Y3)

    ax[2].plot(T,Z4)
    ax[2].plot(T,Z3)

    e1 = localizer.computeError( localizer.EstimatedPath,Invader.TruePathLog)
    e2 = localizer.computeError( Invader.PathLog, Invader.TruePathLog)

    ax[3].plot(T,e1)
    ax[3].plot(T,e2)
    plt.show()






main()

