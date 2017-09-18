
## Code written for MCMC sampling for the Bayesian parameter estimation of 
# post-merger gravitational wave signal of the merger of 2 Neutron stars

### Author: 		Yash Bhargava, IUCAA 
### Last updated: 	September 18, 2017


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf
import argparse 
import emcee as mc
import time
import corner_local as cr



pr = argparse.ArgumentParser()
pr.add_argument("append_text",default="gnh")

pr.add_argument("freq",type=float,default=2.3e3)
pr.add_argument("tau",type=float,default=23.45e-3)
pr.add_argument("amp",type=float,default=0.5)

pr.add_argument("gamma",type=float,default=38.0)
pr.add_argument("xi",type=float,default=-9e2)

pr.add_argument("dim",type=int,default=5)


args = pr.parse_args()

start_time = time.time()			# Timing the operations

date_append = args.append_text		# Text which will be appended to the files 

t = np.linspace(0,0.5, 10000)		# Time duration considered for the signal


seed_freq 	= args.freq
seed_tau 	= args.tau
seed_amp 	= args.amp
seed_gamma	= args.gamma
seed_xi		= args.xi
num_bins = 50


if args.dim==3:
	guess = np.array([2000,0.02,3])				## Format of guess is freq,tau,amp
	labels = ["$f$", r"$\tau$", "$A$"]
	truths=[seed_freq, seed_tau, seed_amp]
	
elif args.dim==2:
	guess = np.array([2000,0.02]) 				## Format of guess is freq,tau
	labels = ["$f$", r"$\tau$"]
	truths=[seed_freq, seed_tau]
elif args.dim==5:
	guess = np.array([2000,0.02,3,100,-1e3])  	## Format of guess is freq,tau,amp,gamma,xi
	labels=["$f$", r"$\tau$", "$A$",r"$\gamma$", r"$\xi$"]
	
	truths=[seed_freq, seed_tau, seed_amp, seed_gamma,seed_xi]




######## Generating the mock signal

sig = mf.f2_sin(t,[seed_freq,seed_tau,seed_amp,seed_gamma,seed_xi])
noise = rnd.normal(0,1,len(t))
samp = sig+noise


## Plotting the mock signal
ax1 = plt.subplot(211)
ax1.plot(t,sig,'.')
ax2 = plt.subplot(212)
ax2.plot(t,samp,'.')
#plt.show()
plt.clf()



dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]

current_time = time.time()

ndim,nwalkers = args.dim,100
iters = 1e3							# Number of iterations each walker goes through


np.savetxt("test_actual_data_{}.txt".format(date_append),np.transpose([t,samp]),fmt=['%.5f','%.5f'],
			header="Freq={}\ttau={}\tamp={}\tgamma={}\txi={}".format(seed_freq,seed_tau,seed_amp,seed_gamma,seed_xi))

pos = [guess+1e-4*rnd.randn(ndim) for i in range (nwalkers)]			# Initialising the walkers

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.f2_sin),threads=4)	# Initialising the sampler
sampler.run_mcmc(pos, iters)																# Running the sampler
print "EMCEE done, time required:\t", time.time()-current_time


## Plotting the samplers

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
for freq in range(np.shape(sampler.chain)[0]):ax1.plot(sampler.chain[freq,:,0],'-k' )
for tau in range(np.shape(sampler.chain)[0]):ax2.plot(sampler.chain[tau,:,1],'-k')
ax1.axhline(seed_freq,color='b')
ax2.axhline(seed_tau,color='b')
ax1.set_ylabel("Freq samples")
ax2.set_xlabel(r"Step number")

ax2.set_ylabel(r"$\tau$ samples")


plt.savefig("mcmc_2d_sample_{}.png".format(date_append))
#plt.show()
plt.clf()


if args.dim>2:
	with open ("mcmc_freq_tau_amp_list_{}.txt".format(date_append),'w') as f: 
		np.savetxt(f,np.transpose([freq_list,tau_list,amp_list]),fmt=['%.3f','%.3f','%.3f'])

samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
print "Shape of samples :\t", np.shape(samples)


if args.dim==3:bins = [num_bins,tau_bins,amp_bins]
elif args.dim==5:bins = [num_bins,tau_bins,amp_bins,num_bins,num_bins]
elif args.dim==2:bins = [num_bins,tau_bins]

### Plotting the triangle plot of the parameters


fig = cr.corner(samples, labels=labels,
                      truths=truths,
                      bins = bins,
                      fill_contours=True,show_titles=True,title_fmt=".3e",#hist_2d_cmap=plt.get_cmap('cool'),
                      quantiles=[0.16,0.84],verbose=True,color='k')
fig.savefig("triangle_actual_data_{}.pdf".format(date_append))




