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
sf = np.fft.rfft(samp)
current_freq_guess = 1000

# Pre defining the parameters for sampling 
iter_number = 1e5
freq_step = 50
multi_count = 0
boost = 0.2


# List of parameters saved in loop
guess_list = []
accept_list = []
freq_step_list = []


for i in range(int(iter_number)):
	h_current = mf.single_sin(t,current_freq_guess)
	hf_current = np.fft.rfft(h_current)
	sh_current = 0.5*( sf*np.conjugate(hf_current)+ hf_current*np.conjugate(sf)-hf_current*np.conjugate(hf_current))
	ll_current = np.real(np.sum(sh_current)*df)
	
	new_freq_guess = rnd.normal(current_freq_guess,freq_step)
	h_new = mf.single_sin(t,new_freq_guess)
	hf_new = np.fft.rfft(h_new)
	sh_new = 0.5*( sf*np.conjugate(hf_new)+ hf_new*np.conjugate(sf)-hf_new*np.conjugate(hf_new))
	ll_new = np.real(np.sum(sh_new)*df)
	
	ll_diff = ll_new-ll_current
	if ll_diff<0: p_accept = np.exp(ll_diff)
	else:p_accept = 1
	accept = rnd.rand()<p_accept
	
	# alternate condition
#	p_accept = (ll_diff)
#	accept = p_accept>=0
	
	if accept:
		current_freq_guess = new_freq_guess
		multi_count = 0
		freq_step *= 1-boost
	else :
		multi_count +=1
	if multi_count==50: 
		freq_step *= 1+boost
		multi_count=0
	guess_list.append(current_freq_guess)
	accept_list.append(ll_diff)
	freq_step_list.append(freq_step)
	
	
	
print max(guess_list), min(guess_list)
print max(accept_list), min(accept_list)

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(freq_step_list	)

ax2.hist(guess_list,1000)
plt.show()


