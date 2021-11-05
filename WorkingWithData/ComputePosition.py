import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import pickle
import lzma

with open('DronePoint.npy', 'rb') as f:
    P_drone = np.load(f,fix_imports=True)

with open('WebcamPoint.npy', 'rb') as f:
    P_webcam = np.load(f,fix_imports=True)

print(np.shape(P_drone))
print(np.shape(P_webcam))

start = 300
endIdx = start + 100

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
DataDict = pickle.load( lzma.open( dir_path+'\VectorData.xz', "rb" ) )
Time = DataDict[ 'Time_Vector' ] 
pos1 = DataDict[ 'Position1' ] 
pos2 = DataDict[ 'Position2' ] 
pos3 = DataDict[ 'Position3' ] 

Time = Time[start:endIdx]

Webcam_Position  = pos1[start:endIdx, :]
Drone_Position  = pos2[start:endIdx, :]
Invader_Position  = pos3[start:endIdx, :]


Q1 = DataDict[ 'Attitude1' ] 
Q2 = DataDict[ 'Attitude2' ] 
Q3 = DataDict[ 'Attitude3' ] 

Webcam_AttitudeQ  = Q1[start:endIdx, :]
Drone_AttitudeQ  = Q2[start:endIdx, :]
Invader_AttitudeQ  = Q3[start:endIdx, :]


ax.plot(pos1[start:endIdx,0], pos1[start:endIdx,1],pos1[start:endIdx,2])
ax.plot(pos2[start:endIdx,0], pos2[start:endIdx,1],pos2[start:endIdx,2])
ax.plot(pos3[start:endIdx,0], pos3[start:endIdx,1],pos3[start:endIdx,2])




fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

WebCamMTX = np.matrix([[1.03777593e+03, 0.00000000e+00, 4.91613361e+02],
                       [0.00000000e+00, 1.05032559e+03, 2.88041810e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
WebCamMTX_inv = np.linalg.inv(WebCamMTX)

DroneCamMTX = np.matrix([[914.47853697,   0.,         473.24589383],
                         [  0.,         913.99081421, 367.21332268],
                         [  0.   ,        0.      ,     1.        ]])


# DroneCamMTX = np.matrix([[1000,   0.,         960/2],
#                          [  0.,         1000, 720/2],
#                          [  0.   ,        0.      ,     1.        ]])


DroneCamMTX_inv = np.linalg.inv(DroneCamMTX)



sz = np.shape(Webcam_Position)
P_WI = np.zeros((sz[0],3))

for i in range(0, len(Webcam_AttitudeQ)):
    re = R.from_euler('zyx', [-90, 90, 0], degrees=True)
    r = R.from_quat(Webcam_AttitudeQ[i,:])
    DCM = r.as_matrix()
    DCM_adj = re.as_matrix()
    Pwrld = WebCamMTX_inv @ np.matrix([P_webcam[i,0],P_webcam[i,1],1.0]).transpose()
    lng = np.linalg.norm(Webcam_Position[0,0] - Invader_Position[0,0])
    Pwrld = lng* Pwrld / np.linalg.norm(Pwrld)
   
    
    Pwrld = DCM_adj @ DCM @ Pwrld

    P_WI[i,:] =  Pwrld.transpose()

P_DI = np.zeros((sz[0],3))

for i in range(0, len(Drone_AttitudeQ)):
    


    
    R3 = R.from_rotvec( np.array([0, 0, -1*np.pi/2])).as_matrix()

    R2 = R.from_rotvec( np.array([0, 1*np.pi/2, 0])).as_matrix()

    R1 = R.from_rotvec(np.array([1, 0, 0])).as_matrix()

    DCM_adj =  R1 @ R2 @ R3

    r = R.from_quat(Drone_AttitudeQ[i,:])
    DCM = r.as_matrix()
    
    Pwrld = DroneCamMTX_inv @ np.matrix([P_drone[i,0],P_drone[i,1],1.0]).transpose()
    lng = np.linalg.norm(Drone_Position[i,:] - Invader_Position[i,:])
    Pwrld =  lng * Pwrld / np.linalg.norm(Pwrld)
    Pwrld = Pwrld 
    
    Pwrld =  Pwrld
   # Pwrld = DCM_adj @ DCM.transpose()  @ np.matrix([1,0,0]).transpose()
    Pwrld = DCM_adj @ DCM  @ Pwrld
    #Pwrld = Pwrld - np.asmatrix(Drone_Position[i,:]).transpose()
    P_DI[i,:] =  Pwrld.transpose()

n = 82
ax1.scatter(Webcam_Position[n,0], Webcam_Position[n,1],Webcam_Position[n,2])
ax1.scatter(Drone_Position[n,0], Drone_Position[n,1],Drone_Position[n,2])
ax1.plot(Invader_Position[n,0], Invader_Position[n,1],Invader_Position[n,2])
ax1.plot(Invader_Position[:,0], Invader_Position[:,1],Invader_Position[:,2])
ax1.plot([Webcam_Position[n,0], Webcam_Position[n,0] + P_WI[n,0]],[Webcam_Position[n,1], Webcam_Position[n,1] + P_WI[n,1]],[Webcam_Position[n,2], Webcam_Position[n,2] + P_WI[n,2]])
ax1.plot(Webcam_Position[:,0] + P_WI[:,0],Webcam_Position[:,1] + P_WI[:,1],Webcam_Position[:,2] + P_WI[:,2])
ax1.plot(Drone_Position[:,0], Drone_Position[:,1], Drone_Position[:,2])

ax1.plot([Drone_Position[n,0], Drone_Position[n,0] + P_DI[n,0]],[Drone_Position[n,1], Drone_Position[n,1] + P_DI[n,1]],[Drone_Position[n,2], Drone_Position[n,2] + P_DI[n,2]])
ax1.plot(Drone_Position[:,0]  + P_DI[:,0],Drone_Position[:,1] + P_DI[:,1],Drone_Position[:,2] + P_DI[:,2])

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(P_DI[:,0], P_DI[:,1] )

ax2[0].set_ylim([-1,1])
ax2[0].set_xlim([-1.333,1.333])

ax2[1].plot(P_drone[:len(P_DI),0], P_drone[:len(P_DI),1] )
ax2[1].set_xlim([0,960])
ax2[1].set_ylim([0,720])
print(ax2[1].get_xlim())



fig3, ax3 = plt.subplots(2,1)
ax3[0].plot(Time, P_DI[:,1] )



ax3[1].plot(Time, P_DI[:,0] )

plt.show()
print(np.shape(Webcam_AttitudeQ))
# with open('WebcamPoint.npy', 'rb') as f:
#     P_webcam = np.load(f,allow_pickle=True)