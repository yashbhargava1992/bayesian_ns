# Bayesian_ns
# Author: Yash Bhargava
# Updated on : Sept 21, 2017


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


The function f2_sin assumes a form of the signal (either GNH3/ALF2) 


# Running the code

The code can be run by following command

python mcmc_actual_data_sim.py -p "alf2_high_2d_jul25" -f 2420 -t 0.01037 -a 10 -g -3467 -x 20000 -d 2

The information about the arguments can be discerned by writing python mcmc_code.py -h 

All the arguments are optional and the default values are those for GNH3 model 
and code computes for 5 dimensions by default.

#Note: 

1)If the sampling is done at dimensions fewer than 5, then the parameters which are known 
must be changed in the definition of the function. 

For e.g. If the assumed signal is of GNH3, and the sampling is done only over freq and tau, 
then rest of the parameters should be manually forced to GNH3 values. This can be done in the file 
"my_function.py" by editing the line which looks like this 

freq,tau,gamma,xi,amp,beta=2.42e3,0.01,-3467,2e4,0.5,0			# alf2

The correct values corresponding to the model can be filled here. 

2) If the expected amplitude is incorrectly filled then the simulation may give larger error bars

For e.g. If the expected amplitude is 0.1 of the noise standard deviation and the value assumed by the function 
(as shown in previous point) is 0.5 then the posterior may not converge to the seed values of freq and tau

