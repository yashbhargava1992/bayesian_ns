import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
pi = np.pi


def damped_sin(t,tau,freq,amp=None,phase=None):
	if amp==None: amp=1
	if phase==None: phase=0
	y = amp*np.exp(-t/tau)*np.sin(2*pi*freq*t+phase)
	#~ y = amp*np.sin(2*pi*freq*t+phase)
	return y
	


t = np.arange(0,10,0.0001)
sig = damped_sin(t,1,1010,10)


for x in range(1):
	noise = rnd.normal(0,1,len(t))
	np.savetxt("sim_data/data_{}.txt".format(x),np.transpose([t,sig+noise]),fmt=['%.5f','%.5f'])

	plt.plot(t,sig+noise,'.',label='Signal+Noise')
	plt.plot(t,sig,'-',label='Signal')
	plt.legend()
	plt.savefig('sim_data/signal_realisation_{}.png'.format(x))
	
	#~ plt.show()
	plt.clf()
