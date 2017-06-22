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

sig = mf.damped_sin(t,2,1010,1)
noise = rnd.normal(0,1,len(t))
samp = sig+noise



dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]

current_time = time.time()
ndim,nwalkers = 3,100

wid = args.width
tau_guess = 2
amp_guess = 1
tau_wid = 1
amp_wid = 0.1

np.savetxt("test_data.txt",np.transpose([t,samp]),fmt=['%.5f','%.5f'])

pos = [[args.guess,tau_guess,amp_guess] + [wid,tau_wid,amp_wid]*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.damped_sin))
sampler.run_mcmc(pos, args.iter)
current_time1 = time.time()
print "EMCEE done, time required:\t", current_time1-current_time

print np.shape(sampler.chain)
freq_list = sampler.chain[:, 50:, 0].flatten()
tau_list = sampler.chain[:, 50:, 1].flatten()
amp_list = sampler.chain[:, 50:, 2].flatten()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.hist(freq_list,100)
ax2.hist(tau_list,100)
ax3.hist(amp_list,100)
hist_2d,x_ed,y_ed = np.histogram2d(freq_list,tau_list,bins=(100,100))
#ax3.hist2d(freq_list,tau_list,100)
X,Y = np.meshgrid(x_ed,y_ed)
#ax3.pcolormesh(X,Y,hist_2d.T)
#plt.colorbar()
print np.shape(freq_list)
with open ("mcmc_freq_tau_amp_list.txt",'w') as f: np.savetxt(f,np.transpose([freq_list,tau_list,amp_list]),fmt=['%.3f','%.3f','%.3f'])
plt.savefig("mcmc_3d.png")
plt.show()
