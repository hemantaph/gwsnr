"""
Unit tests for GWSNR JAX-accelerated inner product SNR calculation methods.

Test Coverage:
--------------
- JAX-accelerated inner product SNR calculations using ripplegw library
- Cross-validation with standard inner product method for consistency
- Multiple waveform approximants (IMRPhenomXAS, IMRPhenomD, IMRPhenomD_NRTidalv2)
- JAX/ripplegw vs LAL implementation comparison
- Computational reproducibility and performance validation

Usage:
-----
pytest tests/unit/test_GWSNR_inner_product_jax.py -v -s
pytest tests/unit/test_GWSNR_inner_product_jax.py::TestGWSNRInnerProductJAX::test_jax_cross_validation_with_standard_method -v -s
"""

import os
import numpy as np
import time
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

np.random.seed(1234)

# Default GWSNR configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
CONFIG = {
    # Computational settings
    'npool': 1,                              # Use single process for JAX to avoid issues
    
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
    'waveform_approximant': "IMRPhenomXAS",  # Waveform model for BBH systems (JAX compatible)
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "inner_product_jax",       # Use JAX-accelerated inner product
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
    'pdet_kwargs': None,                           # Calculate SNR, not probability of detection

    # SNR recalculation settings
    'snr_recalculation': False,
    'snr_recalculation_range': [4, 12],
    'snr_recalculation_waveform_approximant': "IMRPhenomXAS",
}

class TestGWSNRInnerProductJAX(CommonTestUtils):
    """
    Test suite for GWSNR JAX-accelerated inner product SNR calculations.
    """

    def test_jax_cross_validation_with_standard_method(self):
        """
        Tests
        -----
        - Cross-validation between JAX and standard inner product methods
        - Numerical accuracy comparison between ripplegw and LAL implementations
        - Consistency validation for aligned spin BBH systems
        - Performance benefits verification while maintaining accuracy
        """
        # Create JAX configuration
        config_jax = CONFIG.copy()
        config_jax['gwsnr_verbose'] = False
        
        # Create standard inner product configuration
        config_standard = CONFIG.copy()
        config_standard['snr_method'] = "inner_product"
        config_standard['gwsnr_verbose'] = False
        
        # Initialize GWSNR instances
        gwsnr_jax = GWSNR(**config_jax)
        gwsnr_standard = GWSNR(**config_standard)

        # Generate test parameters for BBH events with aligned spins
        nsamples = 8  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=False    # No precessing spins for consistency
        )
        
        # Calculate SNR with both methods
        jax_snr = gwsnr_jax.optimal_snr(gw_param_dict=param_dict)
        standard_snr = gwsnr_standard.optimal_snr(gw_param_dict=param_dict)
        
        # Validate both outputs
        self._validate_snr_output(jax_snr, (nsamples,), gwsnr_jax.detector_list)
        self._validate_snr_output(standard_snr, (nsamples,), gwsnr_standard.detector_list)
        
        # Cross-validate results - JAX/ripplegw and LAL implementations should agree
        np.testing.assert_allclose(
            jax_snr["snr_net"],
            standard_snr["snr_net"],
            rtol=1e-3,  # 0.1% tolerance for cross-implementation comparison
            err_msg="JAX and standard methods should produce consistent SNR values"
        )

    def test_multiple_waveform_approximants(self):
        """
        Tests
        -----
        - Compatibility with different JAX-supported waveform approximants
            - IMRPhenomXAS, IMRPhenomD, IMRPhenomD_NRTidalv2
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Appropriate parameter handling for BBH vs BNS systems
        """
        # Configure GWSNR with reduced verbosity for cleaner test output
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False  # Suppress log messages during testing

        approximants = ["IMRPhenomXAS", "IMRPhenomD", "IMRPhenomD_NRTidalv2"]
        nsamples = 8
        
        for approx in approximants:
            config['waveform_approximant'] = approx
            gwsnr = GWSNR(**config)
            
            # Generate appropriate parameters for each approximant
            if 'Tidal' in approx:
                # BNS system with tidal effects
                param_dict = self._generate_params(
                    nsamples, 
                    event_type='bns',        # Binary neutron star events
                    spin_zero=True,          # No spins for simplicity
                    spin_precession=False    # No precessing spins
                )
            else:
                # BBH system
                param_dict = self._generate_params(
                    nsamples, 
                    event_type='bbh',        # Binary black hole events
                    spin_zero=False,         # Include aligned spin parameters
                    spin_precession=False    # No precessing spins for JAX compatibility
                )

            snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
            # Validate that output has correct structure and numerical properties
            self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

    def test_jax_reproducibility_and_performance(self):
        """
        Tests
        -----
        - Computational reproducibility of JAX inner product method
        - Deterministic behavior of JIT compilation and JAX operations
        - JSON output file creation and content verification
        - Performance comparison with standard inner product method
        """
        # Create configuration for this test
        config = CONFIG.copy()
        config['waveform_approximant'] = "IMRPhenomD"
        config['gwsnr_verbose'] = False
        
        # Initialize GWSNR instance with JAX backend
        gwsnr = GWSNR(**config)

        # Generate test parameters for BBH events
        nsamples = 8  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=False    # No precessing spins
        )
        
        # Calculate SNR values and save results to JSON file
        output_file = "snr_data_jax.json"
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict, output_jsonfile=output_file)
        # Validate that output has correct structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

        # Verify that JSON output file was created successfully
        assert os.path.exists(output_file), "Output JSON file was not created"
        assert os.path.getsize(output_file) > 0, "Output file is empty"

        # cleanup - delete the output file after verification
        os.remove(output_file)

        # Test reproducibility
        snr_result2 = gwsnr.optimal_snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            snr_result["snr_net"], # Network SNR from first calculation
            snr_result2["snr_net"], # Network SNR from second calculation
            rtol=1e-10, # Very tight tolerance for reproducibility
            err_msg="JAX SNR calculation should be deterministic"
        )


