import matplotlib.pyplot as plt
import numpy as np 

x = np.linspace(0, 10, 100)
y = np.cos(x)

fig = plt.figure()

for p in range(50):
    p=3
    updated_x=x+p
    updated_y=np.cos(x)
    plt.plot(updated_x,updated_y)
    plt.draw()  
    x=updated_x
    y=updated_y
    plt.pause(0.01)
    fig.clear()