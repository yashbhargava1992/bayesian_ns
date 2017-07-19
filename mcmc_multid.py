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

seed_freq =  1010
seed_tau = 1
seed_amp = 1



sig = mf.damped_sin(t,[seed_freq,seed_tau,seed_amp])
noise = rnd.normal(0,1,len(t))
samp = sig+noise



dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]

current_time = time.time()
ndim,nwalkers = 3,100

fre_wid = args.width
fre_guess = args.guess
tau_guess = 2
amp_guess = 1
tau_wid = 1
amp_wid = 0.1

np.savetxt("test_data_Jun25.txt",np.transpose([t,samp]),fmt=['%.5f','%.5f'],header="Freq={}\ttau={}\tamp={}".format(seed_freq,seed_tau,seed_amp))

pos = [[fre_guess,tau_guess,amp_guess] + [fre_wid,tau_wid,amp_wid]*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.damped_sin),threads=4)
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
ax1.hist(freq_list,100,normed=True)
#ax2.hist(tau_list,100)
#ax3.hist(amp_list,100)
tau_bins = np.logspace(-3,np.log10(max(tau_list)),100)
amp_bins = np.logspace(-3,np.log10(max(amp_list)),100)
ax2.hist(tau_list,tau_bins,normed=True)
ax3.hist(amp_list,amp_bins,normed=True)
ax1.set_ylabel("Freq hist")

ax2.set_ylabel(r"$\tau$ hist")

ax3.set_ylabel("Amp hist")

ax1.axvline(seed_freq,color='k')
ax2.axvline(seed_tau ,color='k')
ax3.axvline(seed_amp ,color='k')

hist_2d,x_ed,y_ed = np.histogram2d(freq_list,tau_list,bins=(100,100))
#ax3.hist2d(freq_list,tau_list,100)
X,Y = np.meshgrid(x_ed,y_ed)
#ax3.pcolormesh(X,Y,hist_2d.T)
#plt.colorbar()
print np.shape(freq_list)
with open ("mcmc_freq_tau_amp_list_Jun25.txt",'w') as f: np.savetxt(f,np.transpose([freq_list,tau_list,amp_list]),fmt=['%.3f','%.3f','%.3f'])
plt.savefig("mcmc_3d_Jul19.png")
plt.show()
