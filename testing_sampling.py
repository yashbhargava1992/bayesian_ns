import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import my_function as mf


data = np.loadtxt("sim_data/data_0.txt",unpack=True)
t = data[0]
s = data[1]

dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]
noise = rnd.normal(0,1,len(t))

sig = mf.single_sin(t,1010)

samp = sig+noise

	
guess_list,freq_step_list = mf.mcmc_1d(t,samp,1500,1e3)


print max(guess_list), min(guess_list)
#print max(accept_list), min(accept_list)
print max(freq_step_list), min(freq_step_list)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(freq_step_list	)

ax2.hist(guess_list,1000)
plt.savefig("mcmc_test.png")
plt.show()

