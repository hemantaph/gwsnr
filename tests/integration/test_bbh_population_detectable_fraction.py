"""
Integration tests for BBH population detectable fraction calculations.

This test suite validates the selection effect function calculation P(λ|SNR_th)
using GWSNR's probability of detection (Pdet) functionality. The selection effect
represents the detectable fraction of a BBH population given detection thresholds.

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- BBH population detectable fraction calculation using Pdet methods
- Selection effect function validation for astrophysical BBH populations
- Output validation: structure, data types, shapes, and numerical properties
- Performance benchmarking for large population samples
- Astrophysical range validation for detectable fractions

Usage:
-----
pytest tests/integration/test_bbh_population_detectable_fraction.py -v -s
pytest tests/integration/test_bbh_population_detectable_fraction.py::TestBBHSelectionEffect::test_name -v -s
"""

import numpy as np
import time
import os
from gwsnr import GWSNR
from gwsnr.utils import load_json

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
    'spin_resolution': 10,                   # Number of spin grid points
    
    # Waveform generation parameters
    'sampling_frequency': 2048.0,            # Sampling frequency (Hz)
    'waveform_approximant': "IMRPhenomD",    # Waveform model for BBH systems
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "interpolation_aligned_spins",  # Use interpolation with aligned spins
    'interpolator_dir': "./interpolator_pickle", # Directory for saved interpolators
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

class TestBBHSelectionEffect():
    """
    Test suite for BBH population detectable fraction calculations.
    """

    def test_detectable_fraction_bbh(self):
        """
        Tests
        -----
        - BBH population detectable fraction calculation using Pdet methods
        - Selection effect function P(λ|SNR_th) for astrophysical BBH populations
        - Output validation: data types, numerical properties, and astrophysical ranges
        - Performance benchmarking for 10,000 BBH event calculations
        - Integration with astrophysical BBH parameter distributions from LER package
        """
        
        # Create configuration for this test (use existing interpolators for speed)
        config = DEFAULT_CONFIG.copy()
        config['gwsnr_verbose'] = False
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='noncentral_chi2')
        
        # hybrid initialization
        gwsnr = GWSNR(**config)

        # get astrophysical BBH parameters
        # this is generate using ler package
        # Get the path to the current test file directory
        test_dir = os.path.dirname(__file__)
        bbh_params_path = os.path.join(test_dir, 'bbh_gw_params.json')
        gw_params = load_json(bbh_params_path) # 10,000 BBH events

        start = time.time()
        # Pdet calculation
        pdet = gwsnr.pdet(gw_param_dict=gw_params)
        # detectable fraction
        p_lambda = np.mean(pdet['pdet_net'])

        execution_time = time.time() - start

        # print results
        print(f"\nBBH Detectable fraction for {gwsnr.detector_list} network: {p_lambda}")
        print(f"Calculation execution time (seconds): {execution_time}")

        assert isinstance(p_lambda, float), f"expected float64, got {type(p_lambda)}"
        assert np.isfinite(p_lambda), f"Detectable fraction must be finite (no NaN/inf)"
        assert np.isreal(p_lambda), f"Detectable fraction must be real (no complex numbers)"
        assert (p_lambda > 0) and (p_lambda < 1), f"Detectable fraction for {gwsnr.detector_list} network must be in a reasonable astrophysical range, got {p_lambda}"
        assert execution_time < 60, f"Detectable fraction calculation took too long: {execution_time} seconds"

