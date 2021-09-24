import numpy as np
from numpy.random.mtrand import pareto
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal as nvm
#from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

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


def Vel_CostFunc(Pos_Array, Vel_Est, Vel_Prev, delT, parameter=1e-2):
    gObs = g_Vel_Sph_model_obs(Pos_Array, delT)
    X0 = Pos_Array[0,:]
    gEst = g_Vel_Sph_model_Est(X0, Vel_Est, delT)

    u = (gObs-gEst)
    a = (Vel_Est.transpose() - Vel_Prev.transpose()) / delT
    
    J = (1/2) * (u.transpose() @ u) + parameter * (a.transpose() @ a)
    return J, (u.transpose() @ u), (a.transpose() @ a)


def Vel_CostFunc_Grad(Pos_Array, Vel_Est, Vel_Prev, delT, parameter=1e-2):
    h = 1e-6
    H = np.matrix([0.0, 0.0, 0.0])
    gradient = np.matrix([0.0, 0.0, 0.0])

    for i in range(3):
        
        H.itemset(i,h)
        J_p,_,_ =  Vel_CostFunc(Pos_Array, Vel_Est + H, Vel_Prev, delT)
        J_m,_,_ =  Vel_CostFunc(Pos_Array, Vel_Est - H, Vel_Prev, delT)

        delG = (1/(2*h)) * (J_p - J_m)
        gradient.itemset(i,delG)
        
        H.itemset(i,0)

    return(gradient)

def Cam_Estimate_Velocity(Pos_Array, Vel0, Vel_Prev, delT, HyperPara=1e-2, MAXITER = 20, EXITTOL = 1e-4, lr = 1e-2):

    Counter = 0
    ExitCond = False
    V_est = Vel0
    J_arr  = []
    ExitCount = 0

    while (Counter < MAXITER) and ExitCond == False:
        grad = Vel_CostFunc_Grad(Pos_Array, V_est, Vel_Prev, delT, parameter=HyperPara)
        V_est = V_est - lr * grad
        J,_,_ = Vel_CostFunc(Pos_Array, V_est, Vel_Prev, delT, parameter=HyperPara)

        J_arr.append(J[0,0])

        if Counter > 0:
            if abs(J_arr[Counter] - J_arr[Counter-1]) < EXITTOL:
                ExitCount+=1
            else:
                ExitCount = 0

        if ExitCount >= 3:
            ExitCond = True


        Counter+=1

    return V_est, J_arr





delT = 1/10
Vel_Prev = np.matrix([1,-2, 1.5])

Vel_true = np.matrix([0.8,-2.2, 1.51])

X0 = np.matrix([1,1,1])
X1 = X0 + Vel_true * delT


X_arr = np.vstack((X0,X1))

Vel = np.matrix([0.98,-2, 1.50])



# Out_obs = g_Vel_Sph_model_obs(X_arr, delT)

# Out_Est = g_Vel_Sph_model_Est(X0, Vel, delT)

# J = (Out_Est - Out_obs).transpose() @ (Out_Est - Out_obs)

J,ju,ja = Vel_CostFunc(X_arr, Vel, Vel_Prev, delT)

grad = Vel_CostFunc_Grad(X_arr, Vel, Vel_Prev, delT)

V_est, J_arr = Cam_Estimate_Velocity(X_arr, Vel_Prev, Vel_Prev, delT, HyperPara=1e-2, MAXITER = 20, EXITTOL = 1e-8, lr = 1e-5)



print(V_est)

fg, ax= plt.subplots()
ax.plot(J_arr)
plt.show()

#print(Out_obs.transpose())
# print(Out_Est.transpose())

