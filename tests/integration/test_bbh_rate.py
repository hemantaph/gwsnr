"""
Integration tests for BBH rate calculations using GWSNR's hybrid SNR method.

This module tests a simplified pipeline for calculating binary black hole (BBH) merger rates using gravitational wave detection probabilities. The hybrid approach combines interpolation-based SNR calculations with selective inner-product SNR recalculations for improved accuracy.

Rate Calculation:
    R = intrinsic_BBH_merger_rate * (detectable_events / total_events)
    
    Where detectable events are those with network SNR >= threshold (8.0)

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- End-to-end BBH rate calculation workflow
- SNR computation across astrophysical parameter space
- Data validation: types, shapes, finite values, realistic ranges
- Performance benchmarks for efficiency

Usage:
-----
pytest tests/integration/test_bbh_rate.py -v -s
pytest tests/integration/test_bbh_rate.py::TestBBHRateCalculation::test_rate_bbh -v -s
"""

import os
import numpy as np
import time
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

class TestBBHRateCalculation():
    """Tests for BBH rate calculation using GWSNR."""

    intrinsic_BBH_merger_rate = 82226.81960088856 # in yr^-1, not to be confuse with merger rate density in Mpc^-3 yr^-1

    def test_rate_bbh(self):
        """
        Context:
        - Rate equation: R = intrinsic_BBH_merger_rate * (detectable_events / total_events)
        - detectable_events: events with optimal SNR (network) >= snr_th (8.0). 
        - optimal SNR calculation uses hybrid method: interpolation + selective bilby recalculation.

        Tests
        -----
        - rate calculation for BBH events
        - Output validation: data types, shapes, numerical properties
        - Output sanity checks: rate within expected astrophysical range
        - Performance: calculation completes within reasonable time
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

        # get astrophysical BBH parameters
        # this is generate using ler package
        # Get the path to the current test file directory
        test_dir = os.path.dirname(__file__)
        bbh_params_path = os.path.join(test_dir, 'bbh_gw_params.json')
        gw_params = load_json(bbh_params_path) # 10,000 BBH events

        start = time.time()
        # Pdet calculation
        pdet = gwsnr.pdet(gw_param_dict=gw_params)
        # rate calculation
        rate = self.intrinsic_BBH_merger_rate * np.average(pdet["pdet_net"])

        execution_time = time.time() - start

        # print results
        print(f"\nBBH merger rate (yr^-1) for {gwsnr.detector_list} network: {rate}")
        print(f"Rate calculation execution time (seconds): {execution_time}")

        assert isinstance(rate, float), f"expected float64, got {type(rate)}"
        assert np.isfinite(rate), f"rate must be finite (no NaN/inf)"
        assert np.isreal(rate), f"rate must be real (no complex numbers)"
        assert (rate > 100) and (rate<1000), f"rate for {gwsnr.detector_list} network must be in a reasonable astrophysical range, got {rate}"
        assert execution_time < 60, f"rate calculation took too long: {execution_time} seconds"

