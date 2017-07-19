import numpy as np
pi = np.pi


def log_Poisson (D,N):
	#~ return D**N*np.exp(-D)/math.factorial(N)
	return N*np.log(D)-D
	
def damped_sin(t,theta):
	"""
	
	Function to generate a damped sinusoidal signal for any time 't' with supplied 
	frequency and damping timescale. 
	
		
	Input 
	----------------
	t				: Time at which signal is required (in sec)
	theta 			: can consist of following
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
	theta = np.atleast_1d(theta)
	if len(theta)==1: 
		freq = theta
		tau=1
		amp = 1
		t_start = 0
	elif len(theta)==2:
		freq,tau = theta
		amp = 1
		t_start = 0
	elif len(theta)==3:
		freq,tau,amp = theta
		t_start = 0
	else: freq,tau,amp,t_start = theta
	# Defining default value
	#if amp==None: amp=1
	#if t_start==None: t_start=0
	ind = np.where (t>=t_start)
	y = np.zeros(len(t))
	y[ind] = amp*np.exp(-(t[ind]-t_start)/tau)*np.sin(2*pi*freq*(t[ind]-t_start))
	return y

def f2_sin (t,theta):
#~ tau,freq,gamma=38,xi=-9e2,amp=1,beta=0,t_start=0):
	"""
	
	Parameters
	----------------
	t				: Time series over which signal is computed
	theta 			: can consist of following
		tau			    : Damping constant of signal
		freq			: Base frequency of the signal
		amp			    : Amplitude of the signal (Default value=1)
		gamma           : Coefficient of t^2 term inside the sin (Default value=38 for GNH3)
		xi				: Coefficient of t^3 term inside the sin (Default value=-9e2 for GNH3)
		beta			: Phase term inside the sin (Default value=0)
		t_start			: Start point of signal as compared to start of time series (Default value=0, assumes t_start>=t[0])
	
	
	Returns
	-----------------
	y				: Signal comprising of only f2 term in  eq of h+ from Bose et al. 2017
	
	"""
	freq,tau,gamma,xi,amp,beta,t_start=1000,1,38,-9e2,1,0,0
	if len(theta)==2:
		freq,tau = theta
	elif len(theta)==3:
		freq,tau,amp = theta
	elif len(theta)==5:
		freq,tau,amp,gamma,xi = theta 
	elif len(theta)==6:
		freq,tau,amp,gamma,xi,t_start = theta 
		print freq,tau,amp,gamma,xi,t_start 
	
	
	ind = np.where (t>=t_start)
	y = np.zeros(len(t))
	y[ind] = amp*np.exp(-(t[ind]-t_start)/tau)*np.sin(2*pi*(freq*(t[ind]-t_start)+gamma*(t[ind]-t_start)**2+xi*(t[ind]-t_start)**3)+pi*beta)
	
	return y


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

def single_sin(t,theta):
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
	theta = np.atleast_1d(theta)
	freq,amp = 1000,1
	if len(theta)==1: freq=theta
	elif len(theta)==2: freq,amp=theta
	# Defining default value
	y = amp*np.sin(2*pi*freq*(t)) 
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
	

def prior_info_v1 (theta):
	
	if theta > 0: return 0
	return -np.inf


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
	guess_list		: Guesses of the frequency (len of this array is iter_number)
	step_list		: List of the steps throughout the sampling
	
	"""
	dt = t[1]-t[0]
	nyq_freq = 0.5/dt
	xf = np.linspace (0,nyq_freq,len(t)/2)
	df = xf[1]-xf[0]
	sf = np.fft.rfft(samp)
	
	current_guess = init_guess
	multi_count = 0
	
	init_step = step
	# List of parameters saved in loop
	guess_list = []
	accept_list = []
	step_list = []

	for i in range(int(iter_number)):
		h_current = single_sin(t,current_guess)
		ll_current = prior_info_v1(current_guess)+likelihood_estimator(sf,h_current,df)
	
		new_guess = np.random.normal(current_guess,step)
		#while new_freq_guess<0: new_freq_guess = rnd.normal(current_freq_guess,freq_step)
		h_new = single_sin(t,new_guess)
		ll_new = prior_info_v1(new_guess)+likelihood_estimator(sf,h_new,df)

	
		ll_diff = ll_new-ll_current
		if ll_diff<0: p_accept = np.exp(ll_diff)
		else:p_accept = 1
		accept = np.random.rand()<p_accept
	
		# alternate condition
	#	p_accept = (ll_diff)
	#	accept = p_accept>=0
	
		if accept:
			current_guess = new_guess
#~ 			multi_count = 0
#~ 			step *= (1-2*boost)
#~ 		else :
#~ 			multi_count +=1
#~ 		if multi_count==multi_thres and step<=0.5*np.max(guess_list): 
#~ 			step *= 1+boost
#~ 			multi_count=0
#~ 		elif multi_count==multi_thres and step>0.5*np.max(guess_list): 
#~ 			step = init_step
#~ 			multi_count=0
	
		guess_list.append(current_guess)
		accept_list.append(ll_diff)
		step_list.append(step)
	
	return guess_list,step_list

	

def likelihood_estimator (sf,h,df):
	"""
	Input 
	---------------
	s				: FFT of  signal 
	h				: Modelled signal
	df 				: Minimum frequency diff in FFT
	
	Output
	---------------
	ll				: Log likehood of the signal being modelled correctly and noise being gaussian
	
	"""
	hf = np.fft.rfft(h)
	sh = 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
	ll = np.real(np.sum(sh)*df)
	
	return ll



def lnlike_v1 (theta,t,s):
	dt 	= t[1]-t[0]
	nyq_freq = 0.5/dt
	xf 	= np.linspace (0,nyq_freq,len(t)/2)
	df 	= xf[1]-xf[0]
	sf 	= np.fft.rfft(s)
	h 	= single_sin(t,theta)
	hf 	= np.fft.rfft(h)
	sh 	= 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
	ll 	= np.real(np.sum(sh)*df)
	return ll
	

def prior_info_dd (theta):
	amp_min,amp_max = 1e-5,20
	freq_min,freq_max = 1500,3100
	tau_min,tau_max = 1e-3,4e-2
	gamma_min,gamma_max = -4000,500
	xi_min,xi_max = -1000, 1e5
	if len(theta)==2:
		freq,tau = theta
		if ((freq>freq_min and freq<freq_max) and 
			(tau>tau_min and tau < tau_max)): 
			return 0.0

	elif len(theta)==3 : 
		freq 	= theta[0]
		tau 	= theta[1]
		amp 	= theta[2]
		if ((freq>freq_min  and freq<freq_max) and 
			(tau>tau_min and tau <tau_max) and 
			(amp > amp_min and amp < amp_max)): 
				return 0.0
	elif len(theta)==5:
		freq 	= theta[0]
		tau 	= theta[1]
		amp 	= theta[2]
		gamma	= theta[3]
		xi		= theta[4]
		if ((freq>freq_min and freq<freq_max) and 
			(tau>tau_min and tau <tau_max) and
			(gamma>gamma_min and gamma<gamma_max) and
			(xi >xi_min and xi<xi_max) and
			(amp>amp_min and amp<amp_max)):
				return 0.0
	return -np.inf


def lnlike_dd (theta,t,s,func):
	#~ if len(theta)==2:
		#~ freq,tau = theta
		#~ amp = 1
		#~ gamma = None
		#~ xi = None	
	#~ elif len(theta)==3 : 
		#~ freq,tau,amp = theta
		#~ gamma = None
		#~ xi = None
	#~ elif len(theta)==5 : freq,tau,gamma,xi,amp = theta
	dt 	= t[1]-t[0]
	nyq_freq = 0.5/dt
	xf 	= np.linspace (0,nyq_freq,len(t)/2)
	df 	= xf[1]-xf[0]
	sf 	= np.fft.rfft(s)
	h 	= func(t,theta)
	hf 	= np.fft.rfft(h)
	sh 	= 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
	ll 	= np.real(np.sum(sh)*df)
	return ll

def lnprob (theta,t,s,func):
	ll = lnlike_dd(theta,t,s,func)
	lp = prior_info_dd(theta)
	if not np.isfinite(lp): return -np.inf
	return lp + ll



	
	
	



