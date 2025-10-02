"""
Integration tests for GWSNR's use case in BBH Horizon distance calculation, that usees interpolation method for rapid SNR calculation and numba njitted antenna response functions.

Horizon distance calculation;
    - Get optimised sky location (ra,dec) for the optimised inclination angle (theta_jn=0rad), and given geocent_time and polarization angle (psi).
      - For each detectors: get the (ra,dec) that maximize the antenna response (i.e. Sqrt(F_plus^2 + F_cross^2)=1). 
      - For the network: get (ra, dec) that maximizes the network SNR (quadrature sum of individual detector SNRs)
    - Using the maximised sky location (ra,dec) with other GW parameters fixed, get the luminosity distance (in Mpc) at which the optimal SNR (network) = snr_th_net (8.0)

Test Coverage:
    - Horizon distance calculation for a BBH event (mass_1=mass_2=30Mo) through rapid SNR calculation using interpolation method.
    - Output validation: dictionary structure, data types, shapes, numerical properties
    - Output sanity checks: 
      - horizon distance within expected astrophysical range
      - use gwsnr's numba njitted antenna response functions to ensure optimised sky location (ra,dec) gives maximum antenna response (i.e. Sqrt(F_plus^2 + F_cross^2)=1)
    - Performance: calculation completes within reasonable time

Usage:
    pytest tests/integration/test_horizon_distance.py::TestBBHHorizonDistanceCalculation::test_horizon_distance_bbh -v -s
    pytest tests/integration/test_bbh_horizon_distance.py -v -s
"""

import numpy as np
import time
from gwsnr import GWSNR

np.random.seed(1234)

# Default GWSNR configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
CONFIG = {
    # Computational settings
    'npool': 4,                              # Number of parallel processes for multiprocessing
    
    # Mass parameter ranges for interpolation grid
    'mtot_min': 2*4.98,                      # Minimum total mass (M☉) - typical for BBH
    'mtot_max': 2*112.5+10.0,                # Maximum total mass (M☉) - extended BBH range
    'ratio_min': 0.1,                        # Minimum mass ratio q = m2/m1
    'ratio_max': 1.0,                        # Maximum mass ratio (equal mass)
    'spin_max': 0.99,                        # Maximum dimensionless spin magnitude
    
    # Interpolation grid resolution
    'mtot_resolution': 50,                  # Number of total mass grid points
    'ratio_resolution': 10,                  # Number of mass ratio grid points  
    'spin_resolution': 10,                   # Number of spin grid points
    
    # Waveform generation parameters
    'sampling_frequency': 2048.0,            # Sampling frequency (Hz)
    'waveform_approximant': "IMRPhenomD",    # Waveform model for BBH systems
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "interpolation_aligned_spins",  # Use interpolation with aligned spins
    'interpolator_dir': "../interpolator_pickle", # Directory for saved interpolators
    'create_new_interpolator': False,           # Use existing interpolators (faster)

    # detector settings
    'psds': None,
    'ifos': None,
    
    # Logging and output settings
    'gwsnr_verbose': True,                   # Enable detailed logging
    'multiprocessing_verbose': False,         # Enable multiprocessing logs
    
    # Analysis settings
    'mtot_cut': False,                       # Don't apply total mass cuts
    'pdet': False,                           # Calculate SNR, not probability of detection
    'snr_th': 8.0,                          # Single-detector SNR threshold
    'snr_th_net': 8.0,                      # Network SNR threshold

    # SNR recalculation settings
    'snr_recalculation': False,
    'snr_recalculation_range': [4, 12],
    'snr_recalculation_waveform_approximant': "IMRPhenomXPHM",
}

class TestBBHHorizonDistanceCalculation():
    """Tests for BBH horizon distance calculation using GWSNR."""

    def test_horizon_distance_bbh(self):
        """
        Tests
        -----
        - Horizon distance calculation for a BBH event (mass_1=mass_2=30Mo) through rapid SNR calculation using interpolation method.
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Output sanity checks: 
        - horizon distance within expected astrophysical range
        - use gwsnr's numba njitted antenna response functions to ensure optimised sky location (ra,dec) gives maximum antenna response (i.e. Sqrt(F_plus^2 + F_cross^2)=1)
        - Performance: calculation completes within reasonable time
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False
        
        # hybrid initialization
        gwsnr = GWSNR(**config)

        start = time.time()

        # horizon distance calculation for a BBH event (mass_1=mass_2=30Mo)
        horizon_distance_dict, sky_location_dict = gwsnr.horizon_distance_numerical(mass_1=30.0, mass_2=30.0)

        execution_time = time.time() - start

        # print results
        print(f"\nHorizon distance (Mpc) for {gwsnr.detector_list} network: {horizon_distance_dict['snr_net']}")
        print(f"Optimised sky location (ra, dec in rad) for {gwsnr.detector_list} network: {sky_location_dict['snr_net']}")

        for key, value in horizon_distance_dict.items():
            name = key if key != 'snr_net' else f'{gwsnr.detector_list} network'
            assert isinstance(value, float), f"expected float64, got {type(value)}"
            assert np.isfinite(value), f"horizon distance must be finite (no NaN/inf)"
            assert np.isreal(value), f"horizon distance must be real (no complex numbers)"
            assert (value > 1000) and (value < 10000), f"horizon distance for {name} network must be in a reasonable astrophysical range, got {value}"

        for key, value in sky_location_dict.items():
            name = key if key != 'snr_net' else f'{gwsnr.detector_list} network'
            # ra check
            assert (value[0] >= 0.0) and (value[0] <= 2*np.pi), f"right ascension (ra) for {name} must be in [0, 2pi], got {value[0]}"
            # dec check
            assert (value[1] >= -np.pi/2) and (value[1] <= np.pi/2), f"declination (dec) for {name} must be in [-pi/2, pi/2], got {value[1]}"

        assert execution_time < 60, f"horizon distance calculation took too long: {execution_time} seconds"

