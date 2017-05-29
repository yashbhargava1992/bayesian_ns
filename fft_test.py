import numpy as np 
import matplotlib.pyplot as plt


pi=np.pi
f = 20
T = 100
dt=0.01
x = np.arange (0,T,dt) 
y = np.sin (2*pi*f*x)


plt.plot(x,y)
#~ plt.show()
plt.clf()

Nyq_freq = 0.5/dt

yf = np.fft.rfft(y)
xf = np.arange(0,Nyq_freq,1.0/T)
print yf
plt.plot(xf,np.abs(yf)[:-1])
plt.show()
