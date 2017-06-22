import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf


freq,tau = np.loadtxt("mcmc_freq_tau_list.txt",unpack=True)
data = np.loadtxt("test_data.txt",unpack=True)
time = data[0]
sig = data[1]
plt.plot(freq)
#plt.show()
plt.clf()


#hist = np.histogram2d(freq,tau,(100,100))

#plt.imshow(hist[0])
plt.hist2d(freq,tau,bins=50)
plt.colorbar()
#plt.show()
plt.clf()


freq,tau,amp = np.loadtxt("mcmc_freq_tau_list.txt",unpack=True)

#for x
#post = [mf.lnprob([f,t,a],time,sig, mf.damped_sin) for f in freq for t in tau for a in amp)] ## Incorrect syntax
#print np.shape(post)
d = [freq,tau,amp]

post = np.zeros (np.shape(data))

for x,f in enumerate(freq):
	for y,t in enumerate(tau):
		for z,a in enumerate(amp):
			post[x,y,z] = mf.lnprob([f,t,a],time,sig, mf.damped_sin)

marg_a = 

