from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import my_function as mf
import numpy.random as rnd
from matplotlib import rc

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern'],'size':15} )

data = np.loadtxt("sim_data/data_0.txt",unpack=True)
t = data[0]
s = data[1]
#~ x0 = [t,s,0.5,1000]
#~ print log_likelihood (t,n,0.5,1000)
#~ par = op.minimize(log_likelihood,  )
#~ print par

dt = t[1]-t[0]
nyq_freq = 0.5/dt
xf = np.linspace (0,nyq_freq,len(t)/2)
df = xf[1]-xf[0]


###### To do: Get fourier transforms of s and h and get the likelihood
''
seed_tau = 1
seed_freq = 1010
h = mf.damped_sin(t,[seed_freq,seed_tau])
s = mf.damped_sin(t,[seed_freq,seed_tau])
noise = rnd.normal(0,1,len(t))
samp = s+noise
plt.plot(t,s+noise,'.',label='Signal+Noise')
plt.plot(t,s,'-',label='Signal')
plt.legend()
plt.savefig('signal_realisation.png')
sf = np.fft.rfft(samp)
hf = np.fft.rfft(h)

sh = 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))

print np.sum(sh)*df 


#~ plt.plot(xf,np.abs(sf),'.b')
#~ plt.plot(xf,np.abs(hf),'-r')
plt.plot(xf,np.abs(sh[1:]),'.r')
#~ plt.show()
plt.clf()

tau_array = np.logspace (-2,2, 51)
#print tau_array
fre_array = np.linspace (1e3,1.1e3, 51)
#print fre_array

t_all = np.array([])			# List of tau as each point in parameter space is sampled
f_all = np.array([])			# List of freq as each point in parameter space is sampled
l_all = np.array([])			# List of likelihood as each point in parameter space is sampled


for fr in fre_array:
	for ta in tau_array:
		h = mf.damped_sin(t,[fr,ta])
		#~ sf = np.fft.rfft(s)
		hf = np.fft.rfft(h)
		sh = 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
		#ll = np.log10(np.abs(np.sum(sh)*df))
		ll = np.real(np.sum(sh)*df)
		t_all = np.append(t_all,ta)
		f_all = np.append(f_all,fr)
		l_all = np.append(l_all,ll)
#np.savetxt('test.txt',[t_all,f_all,l_all])
bins = [tau_array,fre_array]
print bins
im_base = np.histogram2d(t_all,f_all,bins=bins)
image	= np.histogram2d(t_all,f_all,bins=bins,weights=l_all)

X,Y =np.meshgrid(image[1][:-1],image[2][:-1])

plt.contourf(np.transpose(X),np.transpose(Y),image[0]/im_base [0])
plt.xscale('log')
#~ plt.title(r'$\tau$ vs Freq')
plt.xlabel(r'$\tau$')
plt.ylabel('Freq')
plt.axvline(seed_tau , color='k')
plt.axhline(seed_freq, color='k')
plt.colorbar()
plt.savefig('med_amp_tau_vs_freq.png')
plt.show()
plt.clf()
''


####### Testing whether likelihood method works or not
'''
test_freq1 = 1010
test_freq2 = 200
sig = mf.double_sin(t,test_freq1,test_freq2)
noise = rnd.normal(0,1,len(t))
new_sig = sig
fre_array1 = np.linspace (1e2,1.1e3, 101)
fre_array2 = np.linspace (1e2,1.1e3, 101)

f1_all = np.array([])			# List of freq1 as each point in parameter space is sampled
f2_all = np.array([])			# List of freq2 as each point in parameter space is sampled
l_all = np.array([])			# List of likelihood as each point in parameter space is sampled
sf = np.fft.rfft(new_sig)
plt.plot(t,new_sig,'-',label='Signal')
plt.legend()
plt.savefig('noiseless_far_signal_realisation_freq.png')
plt.clf()
for fr1 in fre_array1:
	for fr2 in fre_array2:
		h = mf.double_sin(t,fr1,fr2)
		#~ sf = np.fft.rfft()
		hf = np.fft.rfft(h)
		sh = 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
		#ll = np.log10(np.abs(np.sum(sh[1:])*df))
		ll = np.real((np.sum(sh[1:])*df))
		f1_all = np.append(f1_all,fr1)
		f2_all = np.append(f2_all,fr2)
		l_all = np.append(l_all,ll)
#~ np.savetxt('test.txt',[t_all,f_all,l_all])
bins = [fre_array1,fre_array2]
print bins
im_base = np.histogram2d(f1_all,f2_all,bins=bins)
image	= np.histogram2d(f1_all,f2_all,bins=bins,weights=l_all)

X,Y =np.meshgrid(image[1][:-1],image[2][:-1])

plt.contourf(np.transpose(X),np.transpose(Y),image[0]/im_base [0])
#~ plt.xscale('log')
plt.title ('Freq vs Freq')
plt.xlabel('Freq')
plt.ylabel('Freq')
plt.colorbar()
plt.savefig('noiseless_far_freq_vs_freq.png')
plt.show()
plt.clf()
'''

######### Test 2

'''
test_freq1 = 1010
test_amp1 = 40
sig = mf.double_sin(t,test_freq1,test_amp1)
noise = rnd.normal(0,1,len(t))
new_sig = sig+noise
fre_array1 = np.linspace (1e3,1.1e3, 51)
amp_array1 = np.logspace (-1,2, 51)

f1_all = np.array([])			# List of freq1 as each point in parameter space is sampled
a1_all = np.array([])			# List of freq2 as each point in parameter space is sampled
l_all = np.array([])			# List of likelihood as each point in parameter space is sampled
sf = np.fft.rfft(new_sig)


for fr1 in fre_array1:
	for am1 in amp_array1:
		h = mf.single_sin(t,fr1,am1)
		#~ sf = np.fft.rfft()
		hf = np.fft.rfft(h)
		sh = 0.5*( sf*np.conjugate(hf)+ hf*np.conjugate(sf)-hf*np.conjugate(hf))
		ll = np.log10(np.abs(np.sum(sh[1:])*df))
		f1_all = np.append(f1_all,fr1)
		a1_all = np.append(a1_all,am1)
		l_all = np.append(l_all,ll)
#~ np.savetxt('test.txt',[t_all,f_all,l_all])
bins = [fre_array1,amp_array1]
print bins
im_base = np.histogram2d(f1_all,a1_all,bins=bins)
image	= np.histogram2d(f1_all,a1_all,bins=bins,weights=l_all)

X,Y =np.meshgrid(image[1][:-1],image[2][:-1])

plt.contourf(np.transpose(X),np.transpose(Y),image[0]/im_base [0])
#~ plt.yscale('log')
plt.colorbar()
plt.xlabel('Freq')
plt.ylabel('Amplitude')
plt.title('Freq vs Amplitude')
plt.savefig('freq_vs_amp.png')
#plt.show()
plt.clf()

'''

