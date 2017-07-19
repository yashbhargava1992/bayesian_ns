import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf
import argparse 
import emcee as mc
import time
from matplotlib import rc

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern'],'size':15} )

pr = argparse.ArgumentParser()
pr.add_argument("guess",type=float,default=500)
pr.add_argument("-i","--iter",type =int, default = 10000)
pr.add_argument("-w","--width",type= float,default = 100)
args = pr.parse_args()

start_time = time.time()
data = np.loadtxt("sim_data/data_0.txt",unpack=True)
t = data[0]
s = data[1]

dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]
noise = rnd.normal(0,1,len(t))


seed_freq = 1010
sig = mf.damped_sin(t,seed_freq)

samp = sig+noise

	
guess_list,freq_step_list = mf.mcmc_1d(t,samp,args.guess,args.iter,args.width)
current_time = time.time()

print "My code done, time required:\t", current_time-start_time
ndim,nwalkers = 1,100

wid = args.width

pos = [args.guess + wid*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.damped_sin),threads=4)
sampler.run_mcmc(pos, args.iter/10)
current_time1 = time.time()

print "EMCEE done, time required:\t", current_time1-current_time

#~ print sampler.chain

print max(guess_list), min(guess_list)
#print max(accept_list), min(accept_list)
print max(freq_step_list), min(freq_step_list)
ax2 = plt.subplot(212)
ax1 = plt.subplot(211)
#~ ax3 = plt.subplot(413)
#~ ax4 = plt.subplot(414)
#~ ax1.plot(freq_step_list	)
#~ ax2.plot(guess_list)
ax1.hist(guess_list,100,normed=True)
ax1.axvline(seed_freq,color='k')
ax2.axvline(seed_freq,color='k')
ax2.hist(sampler.chain[:, 50:, :].reshape((-1, ndim)),100,normed=True)
ax1.set_ylabel('Basic sampling')
ax2.set_ylabel('$emcee$ sampling')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

ax2.set_xlabel('frequency')
plt.savefig("mcmc_test_jul19.png")
plt.show()

