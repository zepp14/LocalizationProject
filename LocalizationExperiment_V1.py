
from UsefulClasses import *

import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
from sklearn.preprocessing import normalize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def main():
    import matplotlib.pyplot as plt
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

        #P2 = [0,0,0.1]
        Speed = 0.75

        SimObj.genLinearPath(DroneList[1], P1, P2, Speed)


        #defender 2 Path
        
        P1 = [14, 0,  0]
        P2 = [7,2, 2]
        Speed = 0.75

        SimObj.genLinearPath(DroneList[2], P1, P2, Speed)


    Invader.PositionCovar = (1e-2)*np.eye(3)
    Def1.PositionCovar = (1e-7)*np.eye(3)
    Def2.PositionCovar = (1e-7)*np.eye(3)
    GenPaths(SimObj, DroneList)

    SimObj.runSim()
    ####
    Def2.LogFreq = 1 #Hz
    Invader.LogFreq = 3 #Hz
    ###


    SimObj.applyDataIntermittence()

    SimObj.applyDataIntermittence()
    Obs = Def1.retrieveRayVect(Invader)
    Obs = Obs + Def1.PathLog
    Obs_n = np.vstack((Def1.PathLog[15,:], Obs[15,:]))

    localizer = LocalizerObject()

    #localizer.ResolveLocationCam(Def1, Invader)
    localizer.ResolveLocationCam_Dual(Def1,Def2, Invader)



    #vel estimation loop
        #Data sets
    Def1_Obs = Def1.retrieveRayVect(Invader)
    Def2_Obs = Def2.retrieveRayVect(Invader)
    RangeMeas = Invader.PathLog

    VelocityEstimate = np.zeros(np.shape(Def1.PathLog))
    PositionEstimate = np.zeros(np.shape(Def1.PathLog))
   
    Tvec = SimObj.TimeVect

        #Def2 Data
    Def2_lastObs = Def2_Obs[0]
    Def2_lastTime = Tvec[0]

        #Range Data
    Range_lastObs = Def2_Obs[0]
    Range_lastTime = RangeMeas[0]


    for i,P in enumerate(Def1.PathLog):
        
        Def2Flag = Def2.AvailFlag[i]
        RangeFlag = Invader.AvailFlag[i]

        if (Def2Flag == 1 and RangeFlag == 1) or Def2Flag == 1:
            VelocityEstimate[i] = Invader.TrueVelocityLog[i+1]
            PositionEstimate[i] = localizer.DualCam3DModel(RangeMeas[i,:], Def1_Obs[i,:], Def2_Obs[i,:], Def1.PathLog[i,:], Def2.PathLog[i,:])

        elif Def2Flag == 0 and RangeFlag == 1:
            PositionEstimate[i] = localizer.Cam3DModel(RangeMeas[i,:], Def1_Obs[i,:], Def1.PathLog[i,:])

            X0 = np.asmatrix(PositionEstimate[i-1])
            X1 = np.asmatrix(PositionEstimate[i])

            Pos_Array = np.vstack((X0,X1))
           
            #VelocityEstimate[i],_ = localizer.Cam_Estimate_Velocity( Pos_Array, np.asmatrix(VelocityEstimate[i-1]), np.asmatrix(VelocityEstimate[i-1]), SimObj.deltaT)
            VelocityEstimate[i] = (1 / SimObj.deltaT ) * (PositionEstimate[i] - PositionEstimate[i-1]) 
            

        elif Def2Flag == 0 and RangeFlag == 0:
            
            X0 = np.asmatrix(Def1_Obs[i-1,:])
            X1 = np.asmatrix(Def1_Obs[i,:])

           
            
            alph = np.linalg.norm(PositionEstimate[i-1] + Invader.TrueVelocityLog[i] * SimObj.deltaT  - Def1.PathLog[i])
            #alph = np.linalg.norm(PositionEstimate[i-1] + VelocityEstimate[i-1] * SimObj.deltaT  - Def1.PathLog[i])
            print(alph)
            PositionEstimate[i] = np.asmatrix(alph) @  normr(X1) + Def1.PathLog[i]
            
            VelocityEstimate[i] = (1 / SimObj.deltaT ) * (PositionEstimate[i] - PositionEstimate[i-1]) 
            
    

            
        else:
            print("Something went wrong; ", Def2Flag, RangeFlag)
    
    #print(PositionEstimate)
    
    #print(Def2.PathLog[:, ~np.isnan(Def2.PathLog).any(axis=0)])
    ##Plot Data
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

    X5 =  PositionEstimate[:,0] 
    Y5 =  PositionEstimate[:,1] 
    Z5 =  PositionEstimate[:,2] 


    VX1 = VelocityEstimate[:,0]
    VY1 = VelocityEstimate[:,1]
    VZ1 = VelocityEstimate[:,2]

    VX2 = Invader.TrueVelocityLog[:,0]
    VY2 = Invader.TrueVelocityLog[:,1]
    VZ2 = Invader.TrueVelocityLog[:,2]


    axs.scatter3D(X1,Y1,Z1)
    axs.scatter3D(X2,Y2,Z2)
    axs.scatter3D(X3,Y3,Z3,s=1)
    #axs.scatter3D(X4,Y4,Z4,s=2)
    axs.scatter3D(X5,Y5,Z5,s=2,c='r')
    axs.legend(["Def1", "Def2", "Invader-Radar", "Invader-Estimate"])
    #axs.plot3D(Obs_n[:,0],Obs_n[:,1],Obs_n[:,2])

    T = SimObj.TimeVect

    
    ax[0].plot(T,X5)
    ax[0].plot(T,X3)

    ax[1].plot(T,Y5)
    ax[1].plot(T,Y3)

    ax[2].plot(T,Z5)
    ax[2].plot(T,Z3)

    fig1, axVel = plt.subplots(3,1)
    fig1, axEr = plt.subplots()
    axVel[0].plot(T,VX1)
    axVel[0].plot(T,VX2)

    axVel[1].plot(T,VY1)
    axVel[1].plot(T,VY2)

    axVel[2].plot(T,VZ1)
    axVel[2].plot(T,VZ2)

    e1 = localizer.computeError( PositionEstimate,Invader.TruePathLog)
    e2 = localizer.computeError( Invader.PathLog, Invader.TruePathLog)

    axEr.plot(T,e1)
    axEr.plot(T,e2)
    axEr.set_title("Error")
    axEr.legend(["Estimator Error", "Simulated Radar Error"])
    plt.show()






main()

