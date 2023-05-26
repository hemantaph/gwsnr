import quintet
import numpy as np
import pylab as plt
q = quintet.Quintet()
# Sample heavy binary black masses and set spins to be zero
nsamples = 100
mass_1 = np.random.uniform(10, 100, nsamples)
mass_2 = np.random.uniform(10, 100, nsamples)
# Sample the extrinsic parameters
iota = np.random.uniform(0, np.pi, nsamples)
psi = np.random.uniform(0, 2*np.pi, nsamples)
luminosity_distance = np.random.uniform(100, 1000, nsamples)
phase = np.random.uniform(0, 2*np.pi, nsamples)
# Sample the time of coalescence and sky location and iota
geocent_time = np.random.uniform(1000, 1e5, nsamples)
ra = np.random.uniform(0, 2*np.pi, nsamples)
dec = np.arccos(np.random.uniform(-1, 1, nsamples)) - np.pi/2
# Compute the SNR
snr = q.snr(mass_1, mass_2, luminosity_distance, iota, psi, phase, geocent_time, ra, dec)
# Plot the SNR distributions for each detector and the total SNR
plt.figure()
plt.hist(snr['H1'], bins=20, histtype='step', label='H1')
plt.hist(snr['L1'], bins=20, histtype='step', label='L1')
plt.hist(snr['V1'], bins=20, histtype='step', label='V1')
plt.hist(snr['opt_snr_net'], bins=20, histtype='step', label='total')
plt.xlabel('SNR')
plt.ylabel('Number of samples')
plt.legend()
plt.show()


# Now compute the same thing but with quintet set at design sensitivity





