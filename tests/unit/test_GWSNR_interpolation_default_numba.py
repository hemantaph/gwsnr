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
pytest tests/unit/test_GWSNR_interpolation_default_numba.py::TestGWSNRInterpolation::test_output_aligned_spins -v -s
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
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        
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
        self._validate_optimal_snr_output(interp_snr, (nsamples,), gwsnr.detector_list)

        # Verify that JSON output file was created successfully
        assert os.path.exists(output_file), "Output JSON file was not created"
        assert os.path.getsize(output_file) > 0, "Output file is empty"

        # delete the output file after verification
        os.remove(output_file)

        # Test computational reproducibility (same inputs should give identical outputs)
        interp_snr2 = gwsnr.optimal_snr(gw_param_dict=param_dict)  # Calculate again with same parameters
        np.testing.assert_allclose(
            interp_snr['optimal_snr_net'],   # Network SNR from first calculation
            interp_snr2['optimal_snr_net'],  # Network SNR from second calculation
            rtol=1e-10,                      # Very tight relative tolerance
            err_msg="SNR calculation is not deterministic"
        )

    def test_invalid_input_handling(self):
        """
        Tests
        -----
        - Robust error handling for invalid input parameters
        - Edge cases: negative masses, NaN/Inf values, empty arrays
        """
        # Configure GWSNR with reduced verbosity for cleaner test output
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config['gwsnr_verbose'] = False  # Suppress log messages during error testing
        
        # Initialize GWSNR instance for error testing
        gwsnr = GWSNR(**config)
        
        # Test Case 1: Negative masses (physically impossible)
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.optimal_snr(gw_param_dict={
                'mass_1': np.array([-30]),      # Negative primary mass (invalid)
                'mass_2': np.array([20]),       # Positive secondary mass
                'luminosity_distance': np.array([400]),  # Valid distance
            })

        # Test Case 2: NaN (Not a Number) values in input
        with pytest.raises((TypeError, ValueError)):
            gwsnr.optimal_snr(gw_param_dict={
                'mass_1': np.array([30, 40]),           # Valid primary masses
                'mass_2': np.array([20, np.nan]),       # NaN in secondary mass (invalid)
                'luminosity_distance': np.array([400, 500]),  # Valid distances
            })

        # Test Case 3: Infinite values in input  
        with pytest.raises((TypeError, ValueError)):
            gwsnr.optimal_snr(gw_param_dict={
                'mass_1': np.array([30, 40]),           # Valid primary masses
                'mass_2': np.array([20, np.inf]),       # Infinite secondary mass (invalid)
                'luminosity_distance': np.array([400, 500]),  # Valid distances
            })

        # Test Case 4: Empty input arrays (no events to process)
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.optimal_snr(gw_param_dict={
                'mass_1': np.array([]),                 # Empty mass array
                'mass_2': np.array([]),                 # Empty mass array  
                'luminosity_distance': np.array([]),    # Empty distance array
            })

    def test_single_event_no_spins(self):
        """
        Tests
        -----
        - SNR calculation for a single event with no spins
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """
        # Configure GWSNR for non-spinning binary analysis
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
            'snr_method': "interpolation_no_spins", # Use no-spins interpolation method
        })

        # Initialize GWSNR with no-spins configuration
        gwsnr = GWSNR(**config)
        
        # Define parameters for a single BBH event (no spin parameters needed)
        param_dict = {
            'mass_1': 30.0,                      # Primary mass (M☉) - typical stellar BH
            'mass_2': 25.0,                      # Secondary mass (M☉) - typical stellar BH
            'luminosity_distance': 400.0,        # Distance (Mpc) - typical for O4 detection
            'theta_jn': 0.5,                     # Inclination angle (rad) - moderate inclination
            'ra': 1.0,                           # Right ascension (rad) - arbitrary sky position
            'dec': 0.2,                          # Declination (rad) - arbitrary sky position
            'psi': 0.3,                          # Polarization angle (rad) - arbitrary orientation
            'phase': 0.1,                        # Coalescence phase (rad) - arbitrary phase
            'geocent_time': 1246527224.169434,   # GPS time (s) - fixed reference time
        }
        
        # Calculate SNR for the single event
        interp_snr = gwsnr.optimal_snr(gw_param_dict=param_dict)
        
        # Validate output structure (expecting single event output shape)
        self._validate_optimal_snr_output(interp_snr, (1,), gwsnr.detector_list)

    def test_custom_input_arguments(self):
        """
        Tests
        -----
        - SNR calculation with custom GWSNR configuration parameters
            - Mass ranges
            - Waveform approximants
            - Detector configurations
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """
        # Generate test parameters for Binary Neutron Star (BNS) events
        nsamples = 20
        param_dict = self._generate_params(
            nsamples, 
            event_type='bns',        # Binary neutron star systems
            spin_zero=True,          # No spins (typical for BNS analysis)
            spin_precession=False    # No precessing spins
        )

        # Create custom configuration optimized for BNS events
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_json')
        config['interpolator_dir'] = gwsnr_dir
        config.update({            
            # Analysis method  
            'snr_method': "interpolation_no_spins", # No-spins method for BNS
            
            # BNS-specific mass ranges (neutron star masses: ~1-3 M☉)
            'mtot_min': 2*1.0,                   # Minimum total mass: 2 M☉ 
            'mtot_max': (2*3.0)*(1.+5.),               # Maximum total mass: (2*m1_max)*(1+z_max)
            'mtot_resolution': 50,                  # Lower resolution for faster testing
            'ratio_resolution': 10,                  # Lower resolution for faster testing
            
            # BNS-optimized waveform settings
            'waveform_approximant': "TaylorF2",  # Post-Newtonian approximant for BNS
            'frequency_domain_source_model': 'lal_binary_neutron_star',  # BNS source model
            'minimum_frequency': 30.0,           # Higher f_min for BNS (better sensitivity)
            
            # Detector configuration
            'ifos': ['CE'],                      # Cosmic Explorer (next-gen detector)
            'mtot_cut': True                     # Apply total mass cuts for BNS range
        })

        # Initialize GWSNR with BNS-optimized configuration
        gwsnr = GWSNR(**config)
        
        # Calculate SNR for BNS test events
        interp_snr = gwsnr.optimal_snr(gw_param_dict=param_dict)

        print(interp_snr)
        
        # Validate output structure and properties
        self._validate_optimal_snr_output(interp_snr, (nsamples,), gwsnr.detector_list)