import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
#hello GITHUB

#DroneObject Class
class DroneObject:

    def __init__(self):
        self.PositionVector = np.array([0, 0, 0]) #X Y Z position Vector
        self.VelocityVector = np.array([0, 0, 0]) #X Y Z position Vector
        self.PositionCovar =  (1e-12)*np.eye(3)
        self.PlannedPath = 0
        self.PathLog = 0

class SimulatorObject:
    def __init__(self, Drones):
        self.DroneArray = Drones
        self.deltaT = 1/10
        self.TimeLength = 13
        

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
        for drone in self.DroneArray:
            drone.PathLog = np.zeros((int(TimeSteps), 3))
            

        for i, T in enumerate(TimeVec):
            for drone in self.DroneArray:
                if i < len(drone.PlannedPath):
                    rnd = self.addRandomNoise(drone, [0, 0, 0])
                    
                    drone.PathLog[i] = drone.PlannedPath[i] + rnd
                else:
                    rnd = self.addRandomNoise(drone, [0, 0, 0])
                    drone.PathLog[i] = drone.PlannedPath[len(drone.PlannedPath)-1] + rnd


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(3,1)
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
        
        P1 = [0, 0,  0]
        P2 = [4,2, 2]
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


    X =  Invader.PathLog[:,0]
    Y =  Invader.PathLog[:,1]
    Z =  Invader.PathLog[:,2]
    
    ax[0].plot(X)
    ax[1].plot(Y)
    ax[2].plot(Z)

    plt.show()


    
    


