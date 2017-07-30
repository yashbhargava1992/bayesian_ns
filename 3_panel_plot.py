import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern'],'size':15} )





data = np.loadtxt("mcmc_freq_tau_amp_list_Jul19.txt",unpack=True)

seed_freq =  1010
seed_tau = 1
seed_amp = 1



num_bins=30

freq_list = data[0]
tau_list = data[1]
amp_list = data[2]
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
plt.subplots_adjust(hspace=0.5)

print min(tau_list),max(tau_list)


ax1.hist(freq_list,num_bins,normed=True)
#ax2.hist(tau_list,100)
#ax3.hist(amp_list,100)
tau_bins = np.logspace(np.log10(min(tau_list)),np.log10(max(tau_list)),num_bins)
amp_bins = np.logspace(np.log10(min(amp_list)),np.log10(max(amp_list)),num_bins)
ax2.hist(tau_list,tau_bins,normed=True)
ax3.hist(amp_list,amp_bins,normed=True)
#ax2.hist(tau_list,num_bins,log=True,normed=True)
#ax3.hist(amp_list,num_bins,log=True,normed=True)
ax1.set_ylabel("Freq hist")
ax1.set_xlabel("$f$")
ax2.set_ylabel(r"$\tau$ hist")
ax2.set_xlabel(r"$\tau$")
ax3.set_ylabel("Amp hist")
ax3.set_xlabel("$A$")
ax1.axvline(seed_freq,color='r',linewidth= 3)
ax2.axvline(seed_tau ,color='r',linewidth= 3)
ax3.axvline(seed_amp ,color='r',linewidth= 3)



plt.show()
