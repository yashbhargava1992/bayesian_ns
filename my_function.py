import numpy as np
pi = np.pi


def log_Poisson (D,N):
	#~ return D**N*np.exp(-D)/math.factorial(N)
	return N*np.log(D)-D
	
def damped_sin(t,tau,freq,amp=None,t_start=None):
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
	if amp==None: amp=1
	if t_start==None: t_start=0
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
	
def double_sin(t,freq1,freq2,amp1=None,amp2=None):
	"""
	
	Function to generate a damped sinusoidal signal for any time 't' with supplied 
	frequency and damping timescale. 
	
		
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
	if amp1==None: amp1=1
	if amp2==None: amp2=0.5
	y = amp1*np.sin(2*pi*freq1*(t)) + amp2*np.sin(2*pi*freq2*(t))
	return y

def single_sin(t,freq1,amp1):
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


