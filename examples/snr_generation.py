"""
This script generates a set of BBH (Binary Black Hole) parameters and calculates the optimal SNR (Signal-to-Noise Ratio) for each set of parameters. The parameters are saved in a json file.
"""
ENABLE_PJRT_COMPATIBILITY=1
import numpy as np
from gwsnr import GWSNR
from gwsnr.utils import save_json

gwsnr = GWSNR(
        npool=int(4),
        mtot_min=2*4.98, # 4.98 Mo is the minimum component mass of BBH systems in GWTC-3
        mtot_max=2*112.5+10.0, # 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
        ratio_min=0.1,
        ratio_max=1.0,
        spin_max=0.99,
        mtot_resolution=200,
        ratio_resolution=20,
        spin_resolution=10,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        frequency_domain_source_model='lal_binary_black_hole',
        minimum_frequency=20.0,
        duration_max=None,
        duration_min=None,
        snr_type="interpolation_no_spins",
        psds=None,
        ifos=None,
        interpolator_dir="./interpolator_pickle",
        create_new_interpolator=False,
        gwsnr_verbose=True,
        multiprocessing_verbose=True,
        mtot_cut=False,
    )

# Setting up the BBH (Binary Black Hole) parameters
# gerneral case, random parameters
# chirp mass can go upto only 95 if f_min=20. to get non zero SNR
nsamples = 10
chirp_mass = np.linspace(5,80,nsamples)
mass_ratio = np.random.uniform(0.2,1,size=nsamples)
mass_1 = (chirp_mass*(1+mass_ratio)**(1/5))/mass_ratio**(3/5)
mass_2 = chirp_mass*mass_ratio**(2/5)*(1+mass_ratio)**(1/5)
total_mass = mass_1+mass_2
mass_ratio = mass_2/mass_1
# Fix luminosity distance
luminosity_distance = 80*np.ones(nsamples)
# Randomly sample everything else:
theta_jn = np.random.uniform(0,2*np.pi, size=nsamples)
ra, dec, psi, phase = np.random.uniform(0,2*np.pi, size=nsamples), np.random.uniform(0,np.pi, size=nsamples), np.random.uniform(0,2*np.pi, size=nsamples), np.random.uniform(0,2*np.pi, size=nsamples)
a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl = 0,0,0,0,0,0 # Zero spin

# create a dictionary
data = {'mass_1': mass_1, 'mass_2': mass_2, 'luminosity_distance': luminosity_distance, 'theta_jn': theta_jn, 'psi': psi, 'phase': phase, 'ra': ra, 'dec': dec}

# Calculate the optimal SNR (Signal-to-Noise Ratio) for each set of parameters
# with interpolation
interp_snr = gwsnr.snr(**data)

# save the SNR and BBH parameters
interp_data = data.copy()
interp_data.update(interp_snr)

# save the dictionary in json format
file_name = './interpolated_snr_data.json'
print(f'saving interpolated SNR results as json file at {file_name}')
save_json(file_name, interp_data);

# Calculate the optimal SNR (Signal-to-Noise Ratio) for each set of parameters
# with inner product
bilby_snr = gwsnr.compute_bilby_snr(**data)

# save the SNR and BBH parameters
bilby_data = data.copy()
bilby_data.update(bilby_snr)

# save the dictionary in json format
file_name = './inner_product_snr_data.json'
print(f'saving inner-product (bilby-like) SNR results as json file at {file_name}')
save_json(file_name, bilby_data);

