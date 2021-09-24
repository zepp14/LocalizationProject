import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
#from sklearn.preprocessing import normalize

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

def Sphere2Cart(X):
    #Convention [radius Azimuth Inclination]
    r = X[0]
    Theta = X[1]
    Phi = X[2]
    x = r*np.sin(Phi) * np.cos(Theta)
    y = r*np.sin(Phi) * np.sin(Theta)
    z = r*np.cos(Phi)

    return [x, y, z]

def g_Vel_Sph_model_obs(Pos_Array, delT):
    #Input 2 by 3 array [X1; X2]
    #return [dAz, dInc]
    Sph_Pos = [Cart2Sphere(P) for P in Pos_Array]
    Sph_Pos = np.array(Sph_Pos)
    
    d_Sph_Pos = (Sph_Pos[1] - Sph_Pos[0]) / delT
    
    return np.matrix([d_Sph_Pos[1], d_Sph_Pos[2]]).transpose()


def g_Vel_Sph_model_Est(Pos_Est, Vel_Est, delT):
    X0 = np.asmatrix(Pos_Est)
    X1 = X0 + Vel_Est * delT
    X_arr = np.vstack((X0,X1))
    Sph_Pos = [Cart2Sphere(P) for P in X_arr]
    Sph_Pos = np.array(Sph_Pos)
    
    d_Sph_Pos = (Sph_Pos[1] - Sph_Pos[0]) / delT

    return np.matrix([d_Sph_Pos[1], d_Sph_Pos[2]]).transpose()

delT = 1/10
Vel = np.matrix([1,-2, 1.5])

X0 = np.matrix([1,1,1])
X1 = X0 + Vel * delT


X_arr = np.vstack((X0,X1))


Vel = np.matrix([1,-2, 1.51])


Out_obs = g_Vel_Sph_model_obs(X_arr, delT)

Out_Est = g_Vel_Sph_model_Est(X0, Vel, delT)

J = (Out_Est - Out_obs).transpose() @ (Out_Est - Out_obs)



print(J)

#print(Out_obs.transpose())
# print(Out_Est.transpose())

