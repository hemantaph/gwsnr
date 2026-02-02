"""
Integration tests for GWSNR BBH horizon distance calculations.

This test suite validates the horizon distance calculation functionality in GWSNR,
testing both numerical and analytical methods for determining the luminosity distance
at which gravitational wave signals reach detection threshold.

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- Numerical horizon distance calculation with optimized sky location
- Analytical horizon distance calculation using interpolation methods
- Sky location optimization for maximum antenna response
- Output validation: structure, data types, shapes, and numerical properties
- Performance benchmarking and reproducibility validation
- Error handling for invalid input types

Usage:
-----
pytest tests/integration/test_bbh_horizon_distance.py -v -s
pytest tests/integration/test_bbh_horizon_distance.py::TestBBHHorizonDistanceCalculation::test_name -v -s
"""

import os
import numpy as np
import time
import pytest
from gwsnr import GWSNR

np.random.seed(1234)

# Default GWSNR configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
DEFAULT_CONFIG = {
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
    'spin_resolution': 5,                   # Number of spin grid points
    
    # Waveform generation parameters
    'sampling_frequency': 2048.0,            # Sampling frequency (Hz)
    'waveform_approximant': "IMRPhenomD",    # Waveform model for BBH systems
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "interpolation_aligned_spins",  # Use interpolation with aligned spins
    'interpolator_dir': "./interpolator_json", # Directory for saved interpolators
    'create_new_interpolator': False,           # Use existing interpolators (faster)
    
    # detector settings
    'psds': None,
    'ifos': None,
    
    # Logging and output settings
    'gwsnr_verbose': True,                   # Enable detailed logging
    'multiprocessing_verbose': False,         # Enable multiprocessing logs
    
    # Analysis settings
    'mtot_cut': False,                       # Don't apply total mass cuts
    'pdet_kwargs': None,                           # Calculate SNR, not probability of detection
}

class TestBBHHorizonDistanceCalculation():
    """
    Test suite for GWSNR BBH horizon distance calculations.
    """

    def test_horizon_distance_bbh_numerical(self):
        """
        Tests
        -----
        - Numerical horizon distance calculation for equal-mass BBH system (30+30 M☉)
        - Sky location optimization for maximum antenna response
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Astrophysical range validation for horizon distances (1-10 kMpc)
        - Reproducibility validation across multiple calculation runs
        - Performance benchmarking for calculation completion time
        - Error handling for array input (should raise TypeError)
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['gwsnr_verbose'] = False
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='noncentral_chi2', include_optimal_snr=False, include_observed_snr=False)
        
        # hybrid initialization
        gwsnr = GWSNR(**config)

        # validation run
        # this also helps initialize any internal njit functions
        horizon_distance_dict_validation, sky_location_dict_validation = gwsnr.horizon_distance_numerical(mass_1=30.0, mass_2=30.0)

        start = time.time()

        # horizon distance calculation for a BBH event (mass_1=mass_2=30Mo)
        horizon_distance_dict, sky_location_dict = gwsnr.horizon_distance_numerical(mass_1=30.0, mass_2=30.0)

        execution_time = time.time() - start

        # print results
        print(f"\nHorizon distance (Mpc) for {gwsnr.detector_list} network: {horizon_distance_dict['optimal_snr_net']}")
        print(f"Optimised sky location (ra, dec in rad) for {gwsnr.detector_list} network: {sky_location_dict['optimal_snr_net']}")

        for key, value in horizon_distance_dict.items():
            name = key if key != 'optimal_snr_net' else f'{gwsnr.detector_list} network'
            assert isinstance(value, float), f"expected float64, got {type(value)}"
            assert np.isfinite(value), f"horizon distance must be finite (no NaN/inf)"
            assert np.isreal(value), f"horizon distance must be real (no complex numbers)"
            assert (value > 1000) and (value < 10000), f"horizon distance for {name} network must be in a reasonable astrophysical range, got {value}"

            # validation for result consistency
            validation_value = horizon_distance_dict_validation[key]
            np.testing.assert_allclose(
                value,
                validation_value,
                rtol=1e-2, # 1% tolerance
                err_msg=f"Horizon distance results mismatch for {key} detector"
            )

        for key, value in sky_location_dict.items():
            name = key if key != 'optimal_snr_net' else f'{gwsnr.detector_list} network'
            # ra check
            assert (value[0] >= 0.0) and (value[0] <= 2*np.pi), f"right ascension (ra) for {name} must be in [0, 2pi], got {value[0]}"
            # dec check
            assert (value[1] >= -np.pi/2) and (value[1] <= np.pi/2), f"declination (dec) for {name} must be in [-pi/2, pi/2], got {value[1]}"
            # # validation for result consistency
            # validation_value = sky_location_dict_validation[key]
            # np.testing.assert_allclose(
            #     value,
            #     validation_value,
            #     rtol=1e-2, # tighter tolerance for sky location
            #     err_msg=f"Sky location results mismatch for {key} detector"
            # )

        assert execution_time < 60, f"horizon distance calculation took too long: {execution_time} seconds"

        # array input should raise TypeError
        with pytest.raises(TypeError):
            horizon_distance_dict, sky_location_dict = gwsnr.horizon_distance_numerical(mass_1=np.array([30.0]), mass_2=np.array([30.0]))

    def test_horizon_distance_bbh_analytical(self):
        """
        Tests
        -----
        - Analytical horizon distance calculation for equal-mass BBH system (30+30 M☉)
        - Rapid SNR calculation using interpolation methods
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Astrophysical range validation for horizon distances (1-10 kMpc)
        - Input consistency validation between scalar and array inputs
        - Performance benchmarking for calculation completion time
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['gwsnr_verbose'] = False
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='noncentral_chi2', include_optimal_snr=False, include_observed_snr=False)
        
        # hybrid initialization
        gwsnr = GWSNR(**config)

        # validation run
        # this also helps initialize any internal njit functions
        horizon_distance_dict_validation = gwsnr.horizon_distance_analytical(mass_1=np.array([30.0]), mass_2=np.array([30.0]))

        start = time.time()

        # horizon distance calculation for a BBH event (mass_1=mass_2=30Mo)
        horizon_distance_dict = gwsnr.horizon_distance_analytical(mass_1=30.0, mass_2=30.0)

        execution_time = time.time() - start

        # print results
        for key, value in horizon_distance_dict.items():
            print(f"\nHorizon distance (Mpc) for {key} detector: {value}")

            assert isinstance(value, np.ndarray), f"expected np.ndarray, got {type(value)}"
            assert np.isfinite(value).all(), f"horizon distance must be finite (no NaN/inf)"
            assert np.isreal(value).all(), f"horizon distance must be real (no complex numbers)"
            assert (value > 1000).all() and (value < 10000).all(), f"horizon distance for {key} detector must be in a reasonable astrophysical range, got {value}"

            # validation for array input and result consistency
            validation_value = horizon_distance_dict_validation[key]
            np.testing.assert_allclose(
                value,
                validation_value,
                rtol=1e-2, # 1% tolerance
                err_msg=f"Horizon distance results mismatch for {key} detector between scalar and array input"
            )

        assert execution_time < 60, f"horizon distance calculation took too long: {execution_time} seconds"

