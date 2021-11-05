import numpy as np
import pickle
import lzma
import cv2
import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
DataDict = pickle.load( lzma.open( dir_path+'\VectorData.xz', "rb" ) )
pos1 = DataDict[ 'Position1' ] 
pos2 = DataDict[ 'Position2' ] 
pos3 = DataDict[ 'Position3' ] 

ax.plot(pos1[200:300:,0], pos1[200:300,1],pos1[200:300,2])
ax.plot(pos2[200:300,0], pos2[200:300,1],pos2[200:300,2])
ax.plot(pos3[1:,0], pos3[1:,1],pos3[1:,2])

#plt.show()
print(np.shape(pos1))



vidName = '\drone.avi'
PathPre = dir_path + vidName
tracker = cv2.TrackerCSRT_create()
vid = cv2.VideoCapture(PathPre)
DronePoint = []

vidName = '\webcam.avi'
PathPre = dir_path + vidName
tracker1 = cv2.TrackerCSRT_create()
vid1 = cv2.VideoCapture(PathPre)
WebcamPoint = []


for i in range(0,250):
    ret, frame = vid.read()
    ret, frame1 = vid1.read()


ret, frame = vid.read()
bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)

ret, frame1 = vid1.read()
bbox = cv2.selectROI(frame1, False)
ok = tracker1.init(frame1, bbox)

while(True):
    try:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.circle(frame, (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), 5, (255,255,0))
            DronePoint.append([(bbox[0]+bbox[2]/2), (bbox[1]+bbox[3]/2)])

        # Display the resulting frame
        cv2.imshow('drone', frame)

        ret, frame1 = vid1.read()
        ok, bbox = tracker1.update(frame1)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame1, p1, p2, (255,0,0), 2, 1)
            cv2.circle(frame1, (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), 5, (255,255,0))
            WebcamPoint.append([(bbox[0]+bbox[2]/2), (bbox[1]+bbox[3]/2)])

        # Display the resulting frame
        cv2.imshow('webcam', frame1)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object

with open('DronePoint.npy', 'wb') as f:
    np.save(f, np.array(DronePoint),allow_pickle=True)

with open('WebcamPoint.npy', 'wb') as f:
    np.save(f, np.array(WebcamPoint),allow_pickle=True)

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()