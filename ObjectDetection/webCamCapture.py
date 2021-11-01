# import the opencv library
import cv2 as cv
import numpy as np
import os
from TrackingAlgorithmObj import update_track
  
# define a video capture object
#vid = cv.VideoCapture(0)
print(os.getcwd())

vidName = '\LightOff.mp4'
PathPre = os.path.dirname(__file__) + vidName
print(os.path.dirname(__file__))
vid = cv.VideoCapture(PathPre)
ret, first_frame = vid.read()
print(first_frame)
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255
prev_map = mask[..., 1]
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv.imshow('frame', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    blurred = gray
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 10, 10, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = 0
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    #mask[..., 2] = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #mask[..., 2] = magnitude / 0.05
   # arrMag = np.reshape(magnitude,(len(magnitude)**2,))
    sz = np.shape(mask[..., 2] )
    arrMag = np.reshape(mask[..., 2],(sz[0]*sz[1],))
    
    P = np.percentile( arrMag, 99.8, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    print(P)
    
    mask1 = cv.threshold(mask[..., 2] ,200,255,cv.THRESH_BINARY)
    map = mask1[1]
    Change = cv.bitwise_and(map,prev_map)

    # Display the resulting frame
    
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)  
    cv.imshow("dense optical flow",  rgb )
    cv.imshow("mask",  Change )
    prev_gray = gray
    prev_map = mask1[1]
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()