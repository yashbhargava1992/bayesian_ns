# Bayesian_ns

The project is to get a posterior estimation code for various parameters describing a post merger signal of NS Ns coalesence. 

The Final code uses emcee package to do the MCMC sampling of the posterior 
The plots are generated using a modified of the corner.py 


The code which generates the signal and computes the posterior is mcmc_actual_data_sim.py

The usage of the code is 
python mcmc_actual_data_sim.py "appending_string" freq tau amp gamma xi dims

where 
freq	: Frequency of the input signal
tau		: Dampening time scale of the input signal
amp		: Amplitude of the signal w.r.t. noise
gamma 	: Derivative of frequency in the input signal
xi	 	: Second Derivative of frequency in the input signal
dims	: Number of dimensions over which the sampling has to be carried out


If dims == 2 then the sampling is done over freq and tau 
and so on


The function f2_sin assumes a form of the signal (either GNH3/ALF2) which can be switched by 
commenting/uncommenting the required lines


