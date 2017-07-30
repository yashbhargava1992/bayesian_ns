
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


#~ pr.add_argument("-i","--iter",type =int, default = 500)
#~ pr.add_argument("-w","--width",type= float,default = 100)
args = pr.parse_args()

start_time = time.time()
#~ data = np.loadtxt("sim_data/data_0.txt",unpack=True)
#~ t = data[0]
#~ s = data[1]

date_append = args.append_text

t = np.linspace(0,0.5, 10000)

#seed_freq 	= 2.42e3
#seed_tau 	= 0.01037
#seed_amp 	= 0.5
#seed_gamma	= -3467
#seed_xi		= 2e4

seed_freq 	= args.freq
seed_tau 	= args.tau
seed_amp 	= args.amp
seed_gamma	= args.gamma
seed_xi		= args.xi
num_bins = 50


if args.dim==3:
	guess = np.array([2000,0.02,3])#,100,-1e3]) ## Format of guess is freq,tau,amp,gamma,xi
	labels = ["$f$", r"$\tau$", "$A$"]
	truths=[seed_freq, seed_tau, seed_amp]
	
elif args.dim==2:
	guess = np.array([2000,0.02])#,100,-1e3]) ## Format of guess is freq,tau,amp,gamma,xi
	labels = ["$f$", r"$\tau$"]
	truths=[seed_freq, seed_tau]
elif args.dim==5:
	guess = np.array([2000,0.02,3,100,-1e3])
	labels=["$f$", r"$\tau$", "$A$",r"$\gamma$", r"$\xi$"]
	
	truths=[seed_freq, seed_tau, seed_amp, seed_gamma,seed_xi]
width = guess/2
iters = 1e3





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
#plt.show()
plt.clf()



dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]

current_time = time.time()
ndim,nwalkers = args.dim,1000



np.savetxt("test_actual_data_{}.txt".format(date_append),np.transpose([t,samp]),fmt=['%.5f','%.5f'],
			header="Freq={}\ttau={}\tamp={}\tgamma={}\txi={}".format(seed_freq,seed_tau,seed_amp,seed_gamma,seed_xi))

pos = [guess+1e-4*rnd.randn(ndim) for i in range (nwalkers)]

sampler = mc.EnsembleSampler(nwalkers,ndim,mf.lnprob,args = (t,samp,mf.f2_sin),threads=4)
sampler.run_mcmc(pos, iters)
current_time1 = time.time()
print "EMCEE done, time required:\t", current_time1-current_time

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
#ax3 = plt.subplot(313)
freq_list = sampler.chain[:, :, 0].flatten()
tau_list = sampler.chain[:, :, 1].flatten()
print np.shape(sampler.chain)
amp_list = sampler.chain[:, 50:, -1].flatten()

#ax1.hist(freq_list,100)
#ax2.hist(tau_list,100)
#ax3.hist(amp_list,100)
freq_bins = np.linspace(np.min(freq_list),np.max(freq_list),num_bins)
tau_bins = np.logspace(-3,np.log10(max(tau_list)),num_bins)
amp_bins = np.logspace(-3,np.log10(max(amp_list)),num_bins)
#ax2.hist(tau_list,tau_bins)
#ax3.hist(amp_list,amp_bins)
for freq in range(np.shape(sampler.chain)[0]):ax1.plot(sampler.chain[freq,:,0],'-k' )
for tau in range(np.shape(sampler.chain)[0]):ax2.plot(sampler.chain[tau,:,1],'-k')
ax1.axhline(seed_freq,color='b')
ax2.axhline(seed_tau,color='b')
ax1.set_ylabel("Freq samples")
ax2.set_xlabel(r"Step number")

ax2.set_ylabel(r"$\tau$ samples")

#ax3.set_ylabel("Amp hist")
freq_list = sampler.chain[:, 50:, 0].flatten()
tau_list = sampler.chain[:, 50:, 1].flatten()
amp_list = sampler.chain[:, 50:, -1].flatten()
hist_2d,x_ed,y_ed = np.histogram2d(freq_list,tau_list,bins=(num_bins,num_bins))
#ax3.hist2d(freq_list,tau_list,100)
X,Y = np.meshgrid(x_ed,y_ed)
#ax3.pcolormesh(X,Y,hist_2d.T)
#plt.colorbar()
#~ print np.shape(freq_list)
if args.dim>2:
	with open ("mcmc_freq_tau_amp_list_{}.txt".format(date_append),'w') as f: np.savetxt(f,np.transpose([freq_list,tau_list,amp_list]),fmt=['%.3f','%.3f','%.3f'])
plt.savefig("mcmc_2d_sample_{}.png".format(date_append))
plt.show()
plt.clf()




samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
print "Shape of samples :\t", np.shape(samples)


if args.dim==3:bins = [num_bins,tau_bins,amp_bins]
elif args.dim==5:bins = [num_bins,tau_bins,amp_bins,num_bins,num_bins]
elif args.dim==2:bins = [num_bins,tau_bins]

fig = cr.corner(samples, labels=labels,
                      truths=truths,
                      bins = bins,
                      fill_contours=True,show_titles=True,title_fmt=".3e",#hist_2d_cmap=plt.get_cmap('cool'),
                      quantiles=[0.16,0.84],verbose=True,color='k')
fig.savefig("triangle_actual_data_{}.pdf".format(date_append))
