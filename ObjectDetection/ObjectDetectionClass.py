import cv2 as cv
import cv2 as cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from statstests import StatCompute
import time
from TrackingAlgorithmObj import update_track
class BodyObject(object):
    def __init__(self):
        self.Position_Estimate = None
        self.Velocity_Estimate = None
        self.MeasurementCount = 0
        self.NewMeasurements = []
        self.GuessedDangerClass = None

class ObjectTracker(object):
    def __init__(self):
        # Initialized Frames

        self.initTime = time.time()
        self.currentTime = time.time()
        self.PrevTime = time.time()

        self.GrayCurrentFrame = None
        self.GrayPreviousFrame = None

        self.CurrentFrame = None
        self.PreviousFrame = None

        self.OF_BitMask_Prev = None
        self.OF_BitMask_Curr = None

        self.PrevMap = None
        self.Magnitude = None
        self.Color2GrayFormat = cv.COLOR_BGR2GRAY

        self.counts = 0
        self.bins = 0

        self.maskedOF = None

        self.PrevTrackedBodies = []
        self.CurrTrackedBodies = []
        self.TestedTrackedBodies = []

        #contours
        self.ContourList = None
        self.CentroidPoints = None

        self.mode = 0

    def firstFrameRet(self, frame):
        self.initTime = time.time()
        self.CurrentFrame = frame
        self.PreviousFrame =  frame

        self.GrayCurrentFrame = self.conditionFrame( frame)
        self.GrayPreviousFrame = self.conditionFrame( frame)

        zeroArray = np.ones(np.shape(frame))
        self.OF_BitMask_Curr = zeroArray

        self.OF_BitMask_Prev = self.OF_BitMask_Curr
        self.OpticalFlowFrameGen()
        self.PrevTime = time.time()
       



    def conditionFrame(self, frame):
        gray = cv.cvtColor(frame, self.Color2GrayFormat )
        szImg =  np.shape( gray )
        blur =  cv.GaussianBlur(gray, (5,5),0)

        return blur


    def RetrieveNewFrame(self, frame):
        #Cycle Prev Frame
        self.PreviousFrame = self.CurrentFrame
        self.GrayPreviousFrame = self.GrayCurrentFrame
        self.OF_BitMask_Curr = self.OF_BitMask_Prev    

        #update current frame
        self.CurrentFrame = frame
        self.GrayCurrentFrame = self.conditionFrame( frame)
        self.OpticalFlowFrameGen()

    #Optical Flow Algo and Mask Gen
    def OpticalFlowFrameGen(self):
        self.CurrTime = time.time()
        prev = self.GrayPreviousFrame
        curr = self.GrayCurrentFrame
        sz = np.shape(prev )
        flow = cv.calcOpticalFlowFarneback(prev, curr, 
                                    None,
                                    0.5, 10, 10, 1, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        self.flow = flow
        
        mask = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        
        arrMag = np.reshape(mask,(sz[0]*sz[1],))
        
        UpperPrctle = np.percentile( arrMag, 99.8, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        



        #print(UpperPrctle)
        (thresh, map)= cv.threshold(mask, UpperPrctle ,255,0)
        
        
        
      
        try:
           
            self.OF_BitMask_Curr = cv.bitwise_and(self.OF_BitMask_Prev, map)
        except:
            pass

        contours, hierarchy = cv.findContours(cv.convertScaleAbs(map) , cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.ContourList = contours

        self.OF_BitMask_Curr = map
        self.OF_BitMask_Prev = map

        #cv.drawContours(self.CurrentFrame, contours, -1, (0,255,0), 3)
        self.getOF_states()
        self.getContourCentroid()
        
        if self.mode == 0:
            pass
            self.deltaT = self.CurrTime - self.PrevTime
            self.InitializtionMode()
        self.PrevTime = self.CurrTime 
    
    #Optical Flow Algo and Mask Gen
    def getOF_states(self):
        #self.counts, self.bins = np.histogram(self.Magnitude)
        pass
    
    def getFrameDifference(self):
        img = cv.bitwise_and(self.GrayPreviousFrame, self.GrayCurrentFrame )
        #img = 255*cv.normalize(self.GrayPreviousFrame - self.GrayCurrentFrame, 0, 255, cv.NORM_MINMAX)
        #print(img)
        return(img)
    
    def getContourCentroid(self):
        Centers = []
        for c in self.ContourList:
            if len(c) > 20 :
                M = cv.moments(c)
                
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                Centers.append([cX,cY])
        self.CentroidPoints = Centers

    def InitializtionMode(self):
        
        self.CurrObs = []
        for c in self.CentroidPoints:
            c_int = np.float64(c)
            BodyTemp = BodyObject()
            BodyTemp.Position_Estimate = np.array(c_int)
            
            BodyTemp.Velocity_Estimate = [self.flow[c[1],c[0],0],self.flow[c[1],c[0],1]] 
            self.CurrObs.append(BodyTemp)

        
        covariance = np.array([[70., 0],[0, 70.]])
        _,_,_  = update_track(self.CurrObs , self.PrevTrackedBodies, self.TestedTrackedBodies , covariance)
        d  = [n.MeasurementCount for n in self.TestedTrackedBodies]
        
        for p in self.TestedTrackedBodies:
            if p.MeasurementCount > 50:
                print(p.MeasurementCount )
                cent = (np.int32(p.Position_Estimate[0]),np.int32(p.Position_Estimate[1]))
                cent2 = (np.int32(p.Position_Estimate[0]) + 5*np.int32(p.Velocity_Estimate[0]) ,np.int32(p.Position_Estimate[1]) + 5*np.int32(p.Velocity_Estimate[1]))
                p.Position_Estimate = np.float64(p.Position_Estimate) + self.deltaT * np.float64(p.Velocity_Estimate)
                
                cv.circle(self.CurrentFrame,  cent, 5, (100, 0, 255), -1)
                cv.rectangle(self.CurrentFrame, (cent[0]-30,cent[1]-30), (cent[0]+30,cent[1]+30), (255,0,0), 2)
                cv.line(self.CurrentFrame,  cent,  cent2 , (255,255,0))
            else:
                
                cent = (np.int32(p.Position_Estimate[0]),np.int32(p.Position_Estimate[1]))
                cent2 = (np.int32(p.Position_Estimate[0]) + 5*np.int32(p.Velocity_Estimate[0]) ,np.int32(p.Position_Estimate[1]) + 5*np.int32(p.Velocity_Estimate[1]))
                p.Position_Estimate = np.float64(p.Position_Estimate) + self.deltaT * np.float64(p.Velocity_Estimate)
                
                cv.circle(self.CurrentFrame,  cent, 5, (100, 0, 255), -1)
                #cv.rectangle(self.CurrentFrame, (cent[0]-30,cent[1]-30), (cent[0]+30,cent[1]+30), (255,0,0), 2)
                cv.line(self.CurrentFrame,  cent,  cent2 , (255,255,0))

            pass

        # for p in self.PrevTrackedBodies:
        #     cv.circle(self.CurrentFrame, (np.int32(p.Position_Estimate[0]),np.int32(p.Position_Estimate[1])), 5, (255, 0, 100), -1)
        #     pass

        self.PrevTrackedBodies = self.CurrObs


    def getClustering(self):

        n_clust = 2
        if len(self.CentroidPoints) > n_clust:
            kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(self.CentroidPoints)
            ClusterCenters = kmeans.cluster_centers_
            for c in ClusterCenters :
                c_int = np.int64(c)
                cv.circle(self.CurrentFrame, (c_int[0],c_int[1]), 5, (255, 0, 255), -1)
            
        else:
            for c in self.CentroidPoints :
                c_int = np.int64(c)
                cv.circle(self.CurrentFrame, (c_int[0],c_int[1]), 5, (255, 0, 255), -1)
        #cv.circle(self.CurrentFrame, (cX, cY), 5, (255, 0, 255), -1)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    tracker = ObjectTracker()
   
    vidName = '\LightOn.mp4'
    PathPre = os.path.dirname(__file__) + vidName
    
    vid = cv.VideoCapture(PathPre)
    #vid = cv.VideoCapture(0)
    ret, first_frame = vid.read()
    tracker.firstFrameRet(first_frame)





    fig = plt.figure()

    #figure, ax = plt.subplots(figsize=(8,6))
    #line1, = ax.hist(tracker.bins[:-1], tracker.bins, weights=tracker.counts)

    #plt.title("Dynamic Plot of sinx",fontsize=25)

    #plt.xlabel("X",fontsize=18)
    #plt.ylabel("sinX",fontsize=18)
    p = 0
    sz = np.shape(first_frame)
   
   
    out = cv.VideoWriter('output.avi', -1, cv2.VideoWriter_fourcc(*'MP4V'), (sz[0], sz[1]))
    while(True):
        #try:
            ret, frame = vid.read()
            

            tracker.RetrieveNewFrame(frame)
            cv.imshow('ColorImage', tracker.CurrentFrame)
            cv.imshow('GrayBlurrImage', tracker.GrayCurrentFrame)
            cv.imshow('OF_Mask_Comb', tracker.OF_BitMask_Curr)
            #out.write(tracker.CurrentFrame)
            #cv.imshow('Diff', tracker.getFrameDifference())
            
            #updated_y = np.cos(x-0.05*p)
            # plt.hist(tracker.MagnitudeArr,bins = 100, range=(0.,1)) 
            # plt.draw()
            # plt.pause(0.001)
            # fig.clear() 
            #line1.set_xdata(x)
           # line1.set_ydata(updated_y)
            #
            #figure.canvas.draw()
            
            #figure.canvas.flush_events()
            p += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        #except Exception as e:
            #print(e)
            
            #break
    
    # After the loop release the cap object
    out.release()
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows() 

