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


freq,tau,amp = np.loadtxt("mcmc_freq_tau_amp_list_Jun25.txt",unpack=True)

#for x
#post = [mf.lnprob([f,t,a],time,sig, mf.damped_sin) for f in freq for t in tau for a in amp)] ## Incorrect syntax
#print np.shape(post)
d = [freq,tau,amp]



ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.hist(freq,100)
print min(tau),max(tau),min(amp),max(amp),np.log10(max(tau)),np.log10(max(amp))
tau_bins = np.logspace(-3,np.log10(max(tau)),100)
amp_bins = np.logspace(-3,np.log10(max(amp)),100)
ax2.hist(tau,tau_bins)
ax3.hist(amp,amp_bins)
ax2.set_xscale('log')
ax3.set_xscale('log')
ax1.set_ylabel("Freq hist")

ax2.set_ylabel(r"$\tau$ hist")

ax3.set_ylabel("Amp hist")
plt.savefig('mcmc_3d_Jun25.png')

#plt.show()
plt.clf()

# The posterior when calculated for complete sample will be huge (100*450)^ndim Thus we need to randomly 

post = [[[0 for f in freq ] for t in tau ] for a in amp ]
for x,f in enumerate(freq):
	for y,t in enumerate(tau):
		for z,a in enumerate(amp):
			post[x,y,z] = mf.lnprob([f,t,a],time,sig, mf.damped_sin)
#post = [[[ mf.lnprob([f,t,a],time,sig, mf.damped_sin) for f in freq ] for t in tau ] for a in amp ]

print np.shape(post)

#for x,f in enumerate(freq):
#	for y,t in enumerate(tau):
#		for z,a in enumerate(amp):
#			post[x,y,z] = mf.lnprob([f,t,a],time,sig, mf.damped_sin)

#marg_a = np.sum(np.exp(post),axis=2)

#plt.hist2d (marg_a[0],marg_a[1],bins=100)
