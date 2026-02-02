"""
Unit Tests for GWSNR Probability of Detection (Pdet) Calculations

This test suite validates the probability of detection functionality in GWSNR,
testing both boolean and probability distribution outputs with different
statistical distributions (non-central chi-squared and Gaussian).

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- Boolean probability of detection with non-central chi-squared and Gaussian distributions
- Probability distribution calculations with different statistical models
- Output validation: structure, data types, shapes, and numerical properties
- Consistent behavior across different distribution types

Usage:
-----
pytest tests/unit/test_GWSNR_pdet.py -v -s
pytest tests/unit/test_GWSNR_pdet.py::TestGWSNRPdet::test_name -v -s
"""

import os
import numpy as np
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

class TestGWSNRPdet(CommonTestUtils):
    """
    Test suite for GWSNR probability of detection (Pdet) calculations.
    """

    def test_output_pdet_obs_noncentral_chi2_bool(self):
        """
        Tests
        -----
        - Boolean probability of detection using non-central chi-squared distribution
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Consistent Pdet calculation for BBH systems with aligned spins
        - Validation of statistical distribution outputs and thresholds
        """

        # Create configuration for this test 
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='noncentral_chi2', include_optimal_snr=False, include_observed_snr=True)
        
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

        # Calculate Pdet values 
        interp_pdet = gwsnr.pdet(gw_param_dict=param_dict)

        # Validate that output has correct structure and numerical properties
        self._validate_pdet_output(interp_pdet, (nsamples,), gwsnr.detector_list, pdet_type='boolean')

        # validate observed SNR values
        self._validate_observed_snr_output(interp_pdet, (nsamples,), gwsnr.detector_list)

    def test_output_pdet_obs_gaussian_bool(self):
        """
        Tests
        -----
        - Boolean probability of detection using Gaussian distribution approximation
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Comparison with non-central chi-squared results for consistency
        - Validation of Gaussian statistical model for Pdet calculations
        """

        # Create configuration for this test 
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='gaussian', include_optimal_snr=False, include_observed_snr=True)
        
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

        # Calculate Pdet values 
        interp_pdet = gwsnr.pdet(gw_param_dict=param_dict)

        # Validate that output has correct structure and numerical properties
        self._validate_pdet_output(interp_pdet, (nsamples,), gwsnr.detector_list, pdet_type='boolean')

        # validate observed SNR values
        self._validate_observed_snr_output(interp_pdet, (nsamples,), gwsnr.detector_list)

    def test_output_pdet_obs_noncentral_chi2_pdf(self):
        """
        Tests
        -----
        - Probability distribution calculation using non-central chi-squared distribution
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Full probability distribution output for statistical analysis
        - Validation of non-central chi-squared PDF calculation accuracy
        """

        # Create configuration for this test 
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='probability_distribution', distribution_type='noncentral_chi2', include_optimal_snr=True, include_observed_snr=False)
        
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

        # Calculate Pdet values 
        interp_pdet = gwsnr.pdet(gw_param_dict=param_dict)

        # Validate that output has correct structure and numerical properties
        self._validate_pdet_output(interp_pdet, (nsamples,), gwsnr.detector_list, pdet_type='probability_distribution')

        # validate optimal SNR values
        self._validate_optimal_snr_output(interp_pdet, (nsamples,), gwsnr.detector_list)

    def test_output_pdet_obs_gaussian_pdf(self):
        """
        Tests
        -----
        - Probability distribution calculation using Gaussian distribution approximation
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Full probability distribution output for statistical analysis
        - Validation of Gaussian PDF approximation for detection statistics
        """

        # Create configuration for this test 
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='probability_distribution', distribution_type='gaussian', include_optimal_snr=True, include_observed_snr=False)
        
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

        # Calculate Pdet values 
        interp_pdet = gwsnr.pdet(gw_param_dict=param_dict)

        # Validate that output has correct structure and numerical properties
        self._validate_pdet_output(interp_pdet, (nsamples,), gwsnr.detector_list, pdet_type='probability_distribution')

        # validate optimal SNR values
        self._validate_optimal_snr_output(interp_pdet, (nsamples,), gwsnr.detector_list)
        
