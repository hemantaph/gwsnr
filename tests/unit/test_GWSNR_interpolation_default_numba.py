"""
Unit Tests for GWSNR Interpolation-based SNR Calculations (Numba Implementation)

This test suite validates the core functionality of GWSNR's interpolation-based
SNR calculation methods, ensuring accuracy, robustness, and reproducibility.

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- SNR calculations for BBH/BNS events with aligned and non-spinning configurations, using IMRPhenomD and TaylorF2 waveform models
- Output validation: structure, data types, shapes, and numerical properties
- Computational reproducibility and deterministic results
- Error handling for invalid inputs (negative masses, NaN/Inf values, empty arrays)
- Custom configurations (mass ranges, waveform approximants, detector setups)
- Detection probability methods (boolean and matched-filter)
- JSON file output generation and integrity

Usage:
-----
pytest tests/unit/test_GWSNR_interpolation_default_numba.py -v -s
pytest tests/unit/test_GWSNR_interpolation_default_numba.py::TestGWSNRInterpolation::test_custom_input_arguments -v -s
"""

import os
import numpy as np
import pytest
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

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

class TestGWSNRInterpolation(CommonTestUtils):
    """
    Test suite for GWSNR interpolation-based SNR calculations.
    """

    def test_output_aligned_spins(self):
        """
        Tests
        -----
        - Aligned spin SNR calculations using interpolation method
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - JSON output file creation and content verification
        - Reproducibility: Deterministic results across multiple runs
        """

        # Create configuration for this test (use existing interpolators for speed)
        config = DEFAULT_CONFIG.copy()
        # gwsnr_dir = os.path.dirname(__file__)
        # gwsnr_dir = os.path.join(gwsnr_dir, './interpolator_pickle')
        config['interpolator_dir'] = '../interpolator_pickle'
        
        # Initialize GWSNR instance with test configuration
        gwsnr = GWSNR(**config)

        # Generate test parameters for BBH events with aligned spins
        nsamples = 20  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=False    # No precessing spins
        )

        # Calculate SNR values and save results to JSON file
        output_file = "snr_data_interpolation.json"
        interp_snr = gwsnr.optimal_snr(gw_param_dict=param_dict, output_jsonfile=output_file)

        # Validate that output has correct structure and numerical properties
        self._validate_snr_output(interp_snr, (nsamples,), gwsnr.detector_list)

        # Verify that JSON output file was created successfully
        assert os.path.exists(output_file), "Output JSON file was not created"
        assert os.path.getsize(output_file) > 0, "Output file is empty"

        # delete the output file after verification
        os.remove(output_file)

        # Test computational reproducibility (same inputs should give identical outputs)
        interp_snr2 = gwsnr.optimal_snr(gw_param_dict=param_dict)  # Calculate again with same parameters
        np.testing.assert_allclose(
            interp_snr["snr_net"],   # Network SNR from first calculation
            interp_snr2["snr_net"],  # Network SNR from second calculation
            rtol=1e-10,                      # Very tight relative tolerance
            err_msg="SNR calculation is not deterministic"
        )

    