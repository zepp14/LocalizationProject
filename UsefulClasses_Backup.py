import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
from sklearn.preprocessing import normalize
from mpl_toolkits import mplot3d


def normr(X):
    x = np.asarray(X)
    if x.ndim > 1:
        A =  [P / np.linalg.norm(P) for P in X]
    else:
        A =  X / np.linalg.norm(X)
    return A

def Cart2Sphere(Xa):
    #Convention [radius Azimuth Inclination]
    X = np.asarray(Xa)

    x = X[0,0]
    y = X[0,1]
    z = X[0,2]

    r = np.sqrt(x**2 + y**2 + z**2)
    Theta =  np.arctan2(y,x)
    Phi = np.arctan2(np.sqrt(x**2 + y**2),z) 
    return [r, Theta, Phi]

def Sphere2Cart(Xa):
    #Convention [radius Azimuth Inclination]
    X = np.asarray(Xa)

    r = X[0,0]
    Theta = X[0,1]
    Phi = X[0,2]
    x = r*np.sin(Phi) * np.cos(Theta)
    y = r*np.sin(Phi) * np.sin(Theta)
    z = r*np.cos(Phi)

    return [x, y, z]


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
        self.LogFreq = 0
        self.AvailFlag = 0

    def retrieveRayVect(self, OtherDrone):
        rnd = [nvm([0, 0, 0], 1e-4*np.eye(3)) for i in  OtherDrone.TruePathLog]
        
        Obs = normalize(OtherDrone.TruePathLog + rnd - self.PathLog)
        #Input Camera Model
        return Obs

    


class SimulatorObject:
    def __init__(self, Drones):
        self.DroneArray = Drones
        self.deltaT = 1/20
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
                    drone.LogFreq = 1/self.deltaT

                else:
                    rnd = self.addRandomNoise(drone, [0, 0, 0])
                    drone.PathLog[i] = drone.PlannedPath[len(drone.PlannedPath)-1] + rnd
                    drone.TruePathLog[i] = drone.PlannedPath[len(drone.PlannedPath)-1] 

                if i > 0:
                    drone.TrueVelocityLog[i] = (drone.TruePathLog[i] - drone.TruePathLog[i-1]) / self.deltaT
                    drone.VelocityLog[i] = (drone.PathLog[i] - drone.PathLog[i-1]) / self.deltaT

    def applyDataIntermittence(self):
        for drone in self.DroneArray:
            logList = drone.PathLog.copy()
            T = 1/drone.LogFreq
            for i,p in enumerate(drone.PathLog):
                if i > 0:
                    if self.TimeVect[i] > T:

                        T = T + 1/drone.LogFreq
                    else:
                        logList[i] = np.array([np.nan,np.nan,np.nan ])
                        T = T
            drone.AvailFlag = 1* ~np.isnan(np.linalg.norm(logList, axis=1))
            
                    
            
            
                
class LocalizerObject:
    def __init__(self):
        self.deltaT = 1/20
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

## new additions

    def g_Vel_Sph_model_obs(self, Pos_Array, delT):
        #Input 2 by 3 array [X1; X2]
        #return [dAz, dInc]
        Sph_Pos = [Cart2Sphere(P) for P in Pos_Array]
        Sph_Pos = np.array(Sph_Pos)
        
        d_Sph_Pos = (Sph_Pos[1] - Sph_Pos[0]) / delT
        
        return np.matrix([d_Sph_Pos[1], d_Sph_Pos[2]]).transpose()


    def g_Vel_Sph_model_Est(self, Pos_Est, Vel_Est, delT):
        X0 = np.asmatrix(Pos_Est)
        X1 = X0 + Vel_Est * delT
        X_arr = np.vstack((X0,X1))
        Sph_Pos = [Cart2Sphere(P) for P in X_arr]
        Sph_Pos = np.array(Sph_Pos)
        
        d_Sph_Pos = (Sph_Pos[1] - Sph_Pos[0]) / delT

        return np.matrix([d_Sph_Pos[1], d_Sph_Pos[2]]).transpose()


    def Vel_CostFunc(self, Pos_Array, Vel_Est, Vel_Prev, delT, parameter=1e-2):
        gObs = self.g_Vel_Sph_model_obs(Pos_Array, delT)
        X0 = Pos_Array[0,:]
        gEst = self.g_Vel_Sph_model_Est(X0, Vel_Est, delT)

        u = (gObs-gEst)
        a = (Vel_Est.transpose() - Vel_Prev.transpose()) / delT
        
        J = (1/2) * (u.transpose() @ u) + parameter * (a.transpose() @ a)
        return J, (u.transpose() @ u), (a.transpose() @ a)


    def Vel_CostFunc_Grad(self, Pos_Array, Vel_Est, Vel_Prev, delT, parameter=1e-2):
        h = 1e-6
        H = np.matrix([0.0, 0.0, 0.0])
        gradient = np.matrix([0.0, 0.0, 0.0])

        for i in range(3):
            
            H.itemset(i,h)
            
            J_p,_,_ =  self.Vel_CostFunc(Pos_Array, Vel_Est + H, Vel_Prev, delT)
            J_m,_,_ =  self.Vel_CostFunc(Pos_Array, Vel_Est - H, Vel_Prev, delT)
            #print('Hello', J_p)
            delG = (1/(2*h)) * (J_p - J_m)
            
            gradient.itemset(i,delG)
            
            H.itemset(i,0)

        return(gradient)

    def Cam_Estimate_Velocity(self, Pos_Array, Vel0, Vel_Prev, delT, HyperPara=1e-2, MAXITER = 20, EXITTOL = 1e-4, lr = 1e-2):

        Counter = 0
        ExitCond = False
        V_est = Vel0
        J_arr  = []
        ExitCount = 0

        while (Counter < MAXITER) and ExitCond == False:
            grad = self.Vel_CostFunc_Grad(Pos_Array, V_est, Vel_Prev, delT, parameter=HyperPara)
            V_est = V_est - lr * grad
            J,_,_ = self.Vel_CostFunc(Pos_Array, V_est, Vel_Prev, delT, parameter=HyperPara)
            
            J_arr.append(J[0,0])
            
            if Counter > 0:
                if abs(J_arr[Counter] - J_arr[Counter-1]) < EXITTOL:
                    ExitCount+=1
                else:
                    ExitCount = 0

            if ExitCount >= 3:
                ExitCond = True


            Counter+=1

        return [V_est[0,0], V_est[0,1],V_est[0,2] ], J_arr


## End of new

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
    Def2.LogFreq = 3 #Hz
    Invader.LogFreq = 1  #Hz
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
            VelocityEstimate[i] = Invader.VelocityLog[i+1]
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

            Pos_Array = np.vstack((X0,X1))

            dY = localizer.g_Vel_Sph_model_obs(Pos_Array, SimObj.deltaT)
            
            dTh = dY[0]
            dPh = dY[1]

            #figure out dY due to relative motion
            V_rel = Def1.VelocityLog[i]
            PosEstInt = PositionEstimate[i-1] + V_rel * SimObj.deltaT
            X1_mod = np.asmatrix(PosEstInt)

            Pos_Array_rel = np.vstack((X0,X1_mod))

            dY_rel = localizer.g_Vel_Sph_model_obs(Pos_Array_rel, SimObj.deltaT)

            dTh_rel = dY_rel[0]
            dPh_rel = dY_rel[1]

            X0_sph = Cart2Sphere(X0)

            X1_Th = X0_sph[1] + dTh * SimObj.deltaT + 1* dTh_rel * SimObj.deltaT 
            X1_phi = X0_sph[2] + dPh * SimObj.deltaT + 1*dPh_rel * SimObj.deltaT 
            #inject r
            Xtru_sph = Cart2Sphere(np.asmatrix(Invader.TruePathLog[i]-Def1.PathLog[i]))

            X1_sph = np.asmatrix([Xtru_sph[0] , X1_Th, X1_phi ])
            #X1_sph = [np.linalg.norm(PositionEstimate[i-1]-Def1.PathLog[i]) , X1_Th, X1_phi ]
            
            Out = Sphere2Cart(X1_sph)
            
            alph = np.linalg.norm(Invader.TruePathLog[i]-Def1.PathLog[i])
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
    
    #axs.plot3D(Obs_n[:,0],Obs_n[:,1],Obs_n[:,2])

    T = SimObj.TimeVect

    
    ax[0].plot(T,X5)
    ax[0].plot(T,X3)

    ax[1].plot(T,Y5)
    ax[1].plot(T,Y3)

    ax[2].plot(T,Z5)
    ax[2].plot(T,Z3)

    fig1, axVel = plt.subplots(3,1)

    axVel[0].scatter(T,VX1)
    axVel[0].plot(T,VX2)

    axVel[1].scatter(T,VY1)
    axVel[1].plot(T,VY2)

    axVel[2].scatter(T,VZ1)
    axVel[2].plot(T,VZ2)

    e1 = localizer.computeError( PositionEstimate,Invader.TruePathLog)
    e2 = localizer.computeError( Invader.PathLog, Invader.TruePathLog)

    ax[3].plot(T,e1)
    ax[3].plot(T,e2)
    plt.show()


    
    


