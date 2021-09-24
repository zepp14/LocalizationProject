import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
from sklearn.preprocessing import normalize
from mpl_toolkits import mplot3d


#DroneObject Class
class DroneObject:

    def __init__(self):
        self.PositionVector = np.array([0, 0, 0]) #X Y Z position Vector
        self.VelocityVector = np.array([0, 0, 0]) #X Y Z position Vector
        self.PositionCovar =  (1e-12)*np.eye(3)
        self.PlannedPath = 0
        self.ObservedRay = 0
        self.PathLog = 0
        self.TrueVelocityLog = 0
        self.VelocityLog = 0
        self.TruePathLog = 0

    def retrieveRayVect(self, OtherDrone):
        rnd = [nvm([0, 0, 0], 1e-4*np.eye(3)) for i in  OtherDrone.TruePathLog]
        
        Obs = normalize(OtherDrone.TruePathLog +rnd - self.PathLog)
        #Input Camera Model
        return Obs


class SimulatorObject:
    def __init__(self, Drones):
        self.DroneArray = Drones
        self.deltaT = 1/10
        self.TimeLength = 13
        self.TimeVect = 0
        

    def genLinearPath(self, DroneObj, P1, P2, Speed):
        P1 = np.reshape(P1,(1,3))
        P2 = np.reshape(P2,(1,3))
        distance = cdist(P1,P2, metric='euclidean')
        TravelTime = distance[0,0] / Speed
        NumofPoints = TravelTime * (1 / self.deltaT)
        print('Travel Time: ', TravelTime, 'sec')
        DroneObj.PlannedPath = np.zeros((int(np.floor(NumofPoints)),3))
        for i,P in enumerate( DroneObj.PlannedPath):
            P = P1 + (P2 - P1)*(i*self.deltaT / TravelTime)
            DroneObj.PlannedPath[i] = P

    def addRandomNoise(self, drone, P):
        TimeSteps = self.TimeLength * (1/self.deltaT)
        return nvm(P, drone.PositionCovar)


    def runSim(self):
        
        TimeSteps = self.TimeLength * (1/self.deltaT)
        
        TimeVec = np.linspace(0,self.TimeLength,int(TimeSteps))
        self.TimeVect = TimeVec
        for drone in self.DroneArray:
            drone.PathLog =     np.zeros((int(TimeSteps), 3))
            drone.TruePathLog = np.zeros((int(TimeSteps), 3))
            drone.VelocityLog = np.zeros((int(TimeSteps), 3))
            drone.TrueVelocityLog = np.zeros((int(TimeSteps), 3))

        for i, T in enumerate(TimeVec):
            for drone in self.DroneArray:
                if i < len(drone.PlannedPath):
                    rnd = self.addRandomNoise(drone, [0, 0, 0])
                    drone.TruePathLog[i] = drone.PlannedPath[i]
                    drone.PathLog[i] = drone.PlannedPath[i] + rnd

                else:
                    rnd = self.addRandomNoise(drone, [0, 0, 0])
                    drone.PathLog[i] = drone.PlannedPath[len(drone.PlannedPath)-1] + rnd
                    drone.TruePathLog[i] = drone.PlannedPath[len(drone.PlannedPath)-1] 

                if i > 0:
                    drone.TrueVelocityLog[i] = (drone.TruePathLog[i] - drone.TruePathLog[i-1]) / self.deltaT
                    drone.VelocityLog[i] = (drone.PathLog[i] - drone.PathLog[i-1]) / self.deltaT  
                
class LocalizerObject:
    def __init__(self):
        self.deltaT = 1/10
        self.maxITER = 25
        self.lr = 1e-2
        self.EstimatedPath = 0
        self.EstimatedVelocity = 0


    def Cam3DModel(self, Xmeas, Xray, Xpos):
        Xm_mat = np.asmatrix(Xmeas).transpose()
        Xr_mat = np.asmatrix(Xray).transpose()
        Xo_mat = np.asmatrix(Xpos).transpose()
        Num_Term = Xr_mat.transpose() @ (Xm_mat-Xo_mat)
        Alph_Star = Num_Term / np.linalg.norm(Xray)**2
       
        Xout = Alph_Star[0,0] * Xray + Xpos
        print(Xout)
        return Xout

    def DualCam3DModel(self, Xmeas, Xray1, Xray2, Xpos1, Xpos2):
        Xm_mat = np.asmatrix(Xmeas).transpose()
        Xr1_mat = np.asmatrix(Xray1).transpose()
        Xr2_mat = np.asmatrix(Xray2).transpose()
        Xo1_mat = np.asmatrix(Xpos1).transpose()
        Xo2_mat = np.asmatrix(Xpos2).transpose()

        XR = np.concatenate((Xr1_mat,Xr2_mat),axis=1)
        XO = np.concatenate((Xo1_mat,Xo2_mat),axis=1)
        B  = np.matrix([-1, 1]).transpose()
        
        Term1 = XR.transpose() @ XR
        Term2 = XR.transpose() @ XO
        A = (np.linalg.inv(Term1) @ Term2) @ B
        
        
        Alph_Star1 = A[0,0]
        Alph_Star2 = -A[1,0]
        
        Xout1 = Alph_Star1 * Xray1 + Xpos1
        Xout2 = Alph_Star2 * Xray2 + Xpos2

        Meas = np.vstack((Xout1,Xout2))
        Xout = np.mean(Meas,axis=0)
        
        print(Xout)
        return Xout

        
    def ResolveLocationCam_Dual(self, Observer1, Observer2, Target):
        CamMeas1 = Observer1.retrieveRayVect(Target)
        CamMeas2 = Observer2.retrieveRayVect(Target)
        RangeMeas = Target.PathLog
        counter = 0
        szMeas = np.shape(RangeMeas)
        self.EstimatedPath = np.zeros(szMeas)
        #go through array
        for i,Point in enumerate(CamMeas1):

            self.EstimatedPath[i,:] = self.DualCam3DModel(RangeMeas[i,:], CamMeas1[i,:],CamMeas2[i,:], Observer1.PathLog[i,:],Observer2.PathLog[i,:])

    def ResolveLocationCam(self, Observer, Target):
        CamMeas = Observer.retrieveRayVect(Target)
        RangeMeas = Target.PathLog
        counter = 0
        szMeas = np.shape(RangeMeas)
        self.EstimatedPath = np.zeros(szMeas)
        #go through array
        for i,Point in enumerate(CamMeas):
            
            self.EstimatedPath[i,:] = self.Cam3DModel(RangeMeas[i,:], CamMeas[i,:], Observer.PathLog[i,:])
           
    def computeError(self, Meas,TruePath):
        error = np.zeros((len(TruePath),1))
        for i,Point in enumerate(TruePath):
            error[i] = np.linalg.norm(Point - Meas[i,:])
        return error

    

if __name__ == "__main__":
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

        #P2 = [0,1,1]
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

    Obs = Def1.retrieveRayVect(Invader)
    Obs = Obs + Def1.PathLog
    Obs_n = np.vstack((Def1.PathLog[15,:], Obs[15,:]))

    localizer = LocalizerObject()

    #localizer.ResolveLocationCam(Def1, Invader)
    localizer.ResolveLocationCam_Dual(Def1,Def2, Invader)

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


    VX1 = Def1.VelocityLog[:,0]
    VY1 = Def1.VelocityLog[:,1]
    VZ1 = Def1.VelocityLog[:,2]

    VX2 = Invader.VelocityLog[:,0]
    VY2 = Invader.VelocityLog[:,1]
    VZ2 = Invader.VelocityLog[:,2]


    axs.plot3D(X1,Y1,Z1)
    axs.plot3D(X2,Y2,Z2)
    axs.plot3D(X3,Y3,Z3)
    axs.plot3D(X4,Y4,Z4)
    
    #axs.plot3D(Obs_n[:,0],Obs_n[:,1],Obs_n[:,2])

    T = SimObj.TimeVect

    # ax[0].plot(T,X4)
    # ax[0].plot(T,X3)

    # ax[1].plot(T,Y4)
    # ax[1].plot(T,Y3)

    # ax[2].plot(T,Z4)
    # ax[2].plot(T,Z3)

    ax[0].plot(T,VX1)
    ax[0].plot(T,VX2)

    ax[1].plot(T,VY1)
    ax[1].plot(T,VY2)

    ax[2].plot(T,VZ1)
    ax[2].plot(T,VZ2)

    e1 = localizer.computeError( localizer.EstimatedPath,Invader.TruePathLog)
    e2 = localizer.computeError( Invader.PathLog, Invader.TruePathLog)

    ax[3].plot(T,e1)
    ax[3].plot(T,e2)
    plt.show()


    
    


