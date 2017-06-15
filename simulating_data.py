import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf



t = np.arange(0,10,0.0001)
sig = mf.damped_sin(t,1,1010)


for x in range(1):
	noise = rnd.normal(0,1,len(t))
	np.savetxt("sim_data/noiseless_data_{}.txt".format(x),np.transpose([t,sig]),fmt=['%.5f','%.5f'])

	#plt.plot(t,sig+noise,'.',label='Signal+Noise')
	plt.plot(t,sig,'-',label='Signal')
	plt.legend()
	plt.savefig('sim_data/noiseless_signal_realisation_{}.png'.format(x))
	
	#~ plt.show()
	plt.clf()
