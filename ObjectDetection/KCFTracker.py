import time, cv2
from threading import Thread
from djitellopy import Tello
import sys,os

vidName = '\LightOn.mp4'
PathPre = os.path.dirname(__file__) + vidName
    
vid = cv2.VideoCapture(PathPre)
#vid = cv2.VideoCapture(0)

#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerCSRT_create()



bbox = (287, 23, 86, 320)
ret, frame = vid.read()
bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    ok, bbox = tracker.update(frame)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()