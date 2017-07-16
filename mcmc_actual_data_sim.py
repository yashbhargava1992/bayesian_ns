
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf
import argparse 
import emcee as mc
import time
import corner_orig as cr



#~ pr = argparse.ArgumentParser()
#~ pr.add_argument("guess",type=float,default=500)
#~ pr.add_argument("-i","--iter",type =int, default = 500)
#~ pr.add_argument("-w","--width",type= float,default = 100)
#~ args = pr.parse_args()

start_time = time.time()
#~ data = np.loadtxt("sim_data/data_0.txt",unpack=True)
#~ t = data[0]
#~ s = data[1]

date_append = "high_amp_gnh_Jul15"

t = np.linspace(0,0.5, 10000)

#seed_freq 	= 2.42e3
#seed_tau 	= 0.01037
#seed_amp 	= 5
#seed_gamma	= -3467
#seed_xi		= 2e4

seed_freq 	= 2.3e3
seed_tau 	= 0.02345
seed_amp 	= 10
seed_gamma	= 38
seed_xi		= -9e2

guess = np.array([2000,0.02,8])#,-3e3,1e4]) ## Format of guess is freq,tau,amp,gamma,xi
width = guess/10
iters = 5e3
num_bins = 50


sig = mf.f2_sin(t,[seed_freq,seed_tau,seed_amp,seed_gamma,seed_xi])
noise = rnd.normal(0,1,len(t))
samp = sig+noise
#print t
#print sig
print np.mean(noise), np.std(noise)
#~ print np.shape(sig)

ax1 = plt.subplot(211)
ax1.plot(t,sig,'.')
ax2 = plt.subplot(212)
ax2.plot(t,samp,'.')
plt.show()
plt.clf()



dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]

current_time = time.time()
ndim,nwalkers = 3,100



np.savetxt("test_actual_data_{}.txt".format(date_append),np.transpose([t,samp]),fmt=['%.5f','%.5f'],
			header="Freq={}\ttau={}\tamp={}\tgamma={}\txi={}".format(seed_freq,seed_tau,seed_amp,seed_gamma,seed_xi))

pos = [guess+width*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.f2_sin),threads=4)
sampler.run_mcmc(pos, iters)
current_time1 = time.time()
print "EMCEE done, time required:\t", current_time1-current_time

print np.shape(sampler.chain)
freq_list = sampler.chain[:, 50:, 0].flatten()
tau_list = sampler.chain[:, 50:, 1].flatten()
amp_list = sampler.chain[:, 50:, -1].flatten()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.hist(freq_list,100)
#ax2.hist(tau_list,100)
#ax3.hist(amp_list,100)
freq_bins = np.linspace(np.min(freq_list),np.max(freq_list),num_bins)
tau_bins = np.logspace(-3,np.log10(max(tau_list)),num_bins)
amp_bins = np.logspace(-3,np.log10(max(amp_list)),num_bins)
ax2.hist(tau_list,tau_bins)
ax3.hist(amp_list,amp_bins)
ax1.set_ylabel("Freq hist")

ax2.set_ylabel(r"$\tau$ hist")

ax3.set_ylabel("Amp hist")

hist_2d,x_ed,y_ed = np.histogram2d(freq_list,tau_list,bins=(num_bins,num_bins))
#ax3.hist2d(freq_list,tau_list,100)
X,Y = np.meshgrid(x_ed,y_ed)
#ax3.pcolormesh(X,Y,hist_2d.T)
#plt.colorbar()
#~ print np.shape(freq_list)
with open ("mcmc_freq_tau_amp_list_{}.txt".format(date_append),'w') as f: np.savetxt(f,np.transpose([freq_list,tau_list,amp_list]),fmt=['%.3f','%.3f','%.3f'])
plt.savefig("mcmc_3d_{}.png".format(date_append))
#~ plt.show()
plt.clf()


samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
print "Shape of samples :\t", np.shape(samples)
fig = cr.corner(samples, labels=["$freq$", r"$\tau$", "$A$"],# r"$\gamma$", r"$\xi$"],
                      truths=[seed_freq, seed_tau, seed_amp],# seed_gamma,seed_xi],
                      bins = [num_bins,tau_bins,amp_bins],#num_bins,num_bins],
                      fill_contours=True,show_title=True,#hist_2d_cmap=plt.get_cmap('cool'),
                      quantiles=[0.05,0.95],verbose=True,color='k')
fig.savefig("triangle_actual_data_{}.pdf".format(date_append))
