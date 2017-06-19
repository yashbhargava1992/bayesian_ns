import numpy as np
pi = np.pi


def log_Poisson (D,N):
	#~ return D**N*np.exp(-D)/math.factorial(N)
	return N*np.log(D)-D
	
def damped_sin(t,tau,freq,amp=1,t_start=0):
	"""
	
	Function to generate a damped sinusoidal signal for any time 't' with supplied 
	frequency and damping timescale. 
	
		
	Input 
	----------------
	t				: Time at which signal is required (in sec)
	tau				: Damping timescale of signal (in sec)
	freq    		: Frequency of oscillation (in Hz)
	amp				: (Optional) Amplitude of signal. Default value 1
	t_start			: (Optional) Time stamp of when the signal begins. Default value 0
	
	Output
	----------------
	y				: Required signal
	
	Note
	----------------
	Due to ordering of arguments the case of optional amplitude and given t_start cannot be used
		
	"""
	
	# Defining default value
	#if amp==None: amp=1
	#if t_start==None: t_start=0
	ind = np.where (t>=t_start)
	y = np.zeros(len(t))
	y[ind] = amp*np.exp(-(t[ind]-t_start)/tau)*np.sin(2*pi*freq*(t[ind]-t_start))
	return y

def log_likelihood_v1 (t_k,N_k,*par):
	
	"""
	Assumes a poisson process for generating counts. Not possible if we have Gaussian noise of zero mean
	"""
	if len(par)==2: tau,freq,amp,phase = par[0],par[1],1,0
	elif len(par)==3: tau,freq,amp,phase = par[0],par[1],par[2],0
	elif len(par)==4: tau,freq,amp,phase = par[0],par[1],par[2],par[3]
	D = damped_sin(t_k,tau,freq,amp,phase)
	ll = np.sum(log_Poisson(D,N_k))
	return -ll
	
def double_sin(t,freq1,freq2,amp1=1,amp2=1):
	"""
	
	Function to generate a sinusoidal signal with two frequencies for any time 't' with supplied 
	frequencies. 
	
		
	Input 
	----------------
	t				: Time at which signal is required (in sec)
	freq    		: Frequency of oscillation (in Hz)
	freq    		: Frequency of oscillation (in Hz)
	amp				: (Optional) Amplitude of signal. Default value 1
	amp				: (Optional) Amplitude of signal. Default value 1
	
	Output
	----------------
	y				: Required signal
	
	Note
	----------------
	Due to ordering of arguments the case of optional amplitude and given t_start cannot be used
		
	"""
	
	# Defining default value
	#if amp1==None: amp1=1
	#if amp2==None: amp2=1
	y = amp1*np.sin(2*pi*freq1*(t)) + amp2*np.sin(2*pi*freq2*(t))
	return y

def single_sin(t,freq1,amp1=1):
	"""
	
	Function to generate a damped sinusoidal signal for any time 't' with supplied 
	frequency and damping timescale. 
	
		
	Input 
	----------------
	t				: Time at which signal is required (in sec)
	freq    		: Frequency of oscillation (in Hz)
	amp				: (Optional) Amplitude of signal. Default value 1
	
	Output
	----------------
	y				: Required signal
	
	Note
	----------------
	Due to ordering of arguments the case of optional amplitude and given t_start cannot be used
		
	"""
	
	# Defining default value
	y = amp1*np.sin(2*pi*freq1*(t)) 
	return y


def mcmc_1d(t,samp,init_guess = 2000,iter_number = 1e4,step = 100, multi_thres = 50,boost = 0.1):
	
	"""
	Input
	--------------
	t				: X axis of the data
	samp			: Y axis of the data
	init_guess 		: Initial guess of frequency (Default value=2000)
	iter_number 	: Number of sampling to be done (Default value = 1e4)
	step 			: Width of the gaussian around the guess (Default value = 100)
	multi_thres 	: Number of allowed steps to be fixed at the same value (Default value = 50)
	boost 			: The fraction by which the step is modified if multi_thres is reached (Default value = 0.1)
	
	Output 
	--------------
	guess_list		: GUesses of the frequency (len of this array is iter_number)
	step_list		: List of the steps throughout the sampling
	
	"""
	dt = t[1]-t[0]
	nyq_freq = 0.5/dt
	xf = np.linspace (0,nyq_freq,len(t)/2)
	df = xf[1]-xf[0]
	sf = np.fft.rfft(samp)
	
	current_guess = init_guess
	multi_count = 0
	# List of parameters saved in loop
	guess_list = []
	accept_list = []
	step_list = []


	for i in range(int(iter_number)):
		h_current = single_sin(t,current_guess)
		hf_current = np.fft.rfft(h_current)
		sh_current = 0.5*( sf*np.conjugate(hf_current)+ hf_current*np.conjugate(sf)-hf_current*np.conjugate(hf_current))
		ll_current = np.real(np.sum(sh_current)*df)
	
		new_guess = np.random.normal(current_guess,step)
		#while new_freq_guess<0: new_freq_guess = rnd.normal(current_freq_guess,freq_step)
		h_new = single_sin(t,new_guess)
		hf_new = np.fft.rfft(h_new)
		sh_new = 0.5*( sf*np.conjugate(hf_new)+ hf_new*np.conjugate(sf)-hf_new*np.conjugate(hf_new))
		ll_new = np.real(np.sum(sh_new)*df)
	
		ll_diff = ll_new-ll_current
		if ll_diff<0: p_accept = np.exp(ll_diff)
		else:p_accept = 1
		accept = np.random.rand()<p_accept
	
		# alternate condition
	#	p_accept = (ll_diff)
	#	accept = p_accept>=0
	
		if accept:
			current_guess = new_guess
			multi_count = 0
			step *= (1-2*boost)
		else :
			multi_count +=1
		if multi_count==multi_thres and step<=0.5*np.max(guess_list): 
			step *= 1+boost
			multi_count=0
		elif multi_count==multi_thres and step>0.5*np.max(guess_list): 
			step = 10
			multi_count=0
	
		guess_list.append(current_guess)
		accept_list.append(ll_diff)
		step_list.append(step)
	
	return guess_list,step_list

	
	
	
	



