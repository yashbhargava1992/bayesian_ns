import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf
import argparse 
import emcee as mc
import time

pr = argparse.ArgumentParser()
pr.add_argument("guess",type=float,default=500)
pr.add_argument("-i","--iter",type =int, default = 500)
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

current_time = time.time()
ndim,nwalkers = 2,100

wid = args.width

pos = [[args.guess,2] + [wid,1]*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,s))
sampler.run_mcmc(pos, args.iter)
current_time1 = time.time()
print "EMCEE done, time required:\t", current_time1-current_time

print np.shape(sampler.chain)
freq_list = sampler.chain[:, 50:, 0].flatten()
tau_list = sampler.chain[:, 50:, 1].flatten()

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.hist(freq_list,100)
ax2.hist(tau_list,100)

hist_2d = np.histogram2d(freq_list,tau_list)
ax3.imshow(hist_2d[0])
print np.shape(freq_list)
with open ("mcmc_freq_tau_list.txt",'w') as f: np.savetxt(f,[freq_list,tau_list])
plt.savefig("mcmc_2d.png")
plt.show()
