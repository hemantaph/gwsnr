"""
Unit Tests for GWSNR JAX-Accelerated Interpolation Backend

Test Coverage:
- JAX aligned spins interpolation: "interpolation_aligned_spins_jax" backend, using IMRPhenomD waveform model
- JAX no spins interpolation: "interpolation_no_spins_jax" backend, using IMRPhenomD waveform model
- Output validation: dictionary structure, data types, shapes, numerical properties
- Reproducibility testing: Deterministic output verification after JIT compilation
- Cross-validation: Comparison with standard Numba backend results

Usage:
pytest tests/unit/test_GWSNR_interpolation_jax.py -v -s
pytest tests/unit/test_GWSNR_interpolation_jax.py::TestGWSNRInterpolationJAX::test_name -v -s
"""

import numpy as np
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

np.random.seed(1234)

# JAX-specific GWSNR configuration dictionary for all tests
# Optimized for JAX backend performance with reduced resolution for faster testing
JAX_CONFIG = {
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
    
    # JAX-specific SNR calculation settings
    'snr_method': "interpolation_aligned_spins_jax", # Default to JAX aligned spins backend
    'interpolator_dir': "../interpolator_pickle", # Directory for saved interpolators
    'create_new_interpolator': False,           # Use existing interpolators (faster)

    # detector settings
    'psds': None,  # default to None to use built-in PSDs
    'ifos': None,  # default to None to use built-in detectors
    
    # Logging and output settings
    'gwsnr_verbose': True,                   # Enable detailed logging
    'multiprocessing_verbose': False,         # Enable multiprocessing logs
    
    # Analysis settings
    'mtot_cut': False,                       # Don't apply total mass cuts
    'pdet_kwargs': None,                           # Calculate SNR, not probability of detection
}

class TestGWSNRInterpolationJAX(CommonTestUtils):
    """Test suite for GWSNR JAX-accelerated interpolation backends."""

    def test_jax_aligned_spins_interpolation(self):
        """
        Tests
        -----
        - JAX aligned spins interpolation: "interpolation_aligned_spins_jax" backend
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Reproducibility testing: Deterministic output verification after JIT compilation
        - Cross-validation: Comparison with standard Numba backend results
        """
        config = JAX_CONFIG.copy()
        config.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
        })
        
        # Initialize GWSNR with JAX configuration
        gwsnr = GWSNR(**config)
        
        # Generate parameters for BBH events with aligned spins
        nsamples = 20
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',
            spin_zero=False,
            spin_precession=False
        )
        
        # Calculate SNR using JAX backend  
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        
        # Validate JAX output structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)
        
        # Test JAX reproducibility (JIT compilation should be deterministic)
        snr_result2 = gwsnr.optimal_snr(gw_param_dict=param_dict)  # Second call uses compiled function
        np.testing.assert_allclose(
            snr_result["snr_net"],   # Network SNR from first calculation
            snr_result2["snr_net"],  # Network SNR from second calculation (JIT compiled)
            rtol=1e-10,                      # Very tight tolerance for JAX determinism
            err_msg="JAX backend should be deterministic after JIT compilation"
        )
        
        ################################################
        # Cross-validation with standard Numba backend #
        ################################################
        # Standard Numba backend for comparison
        config_numba = JAX_CONFIG.copy()  # Use same base config
        config_numba.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
            'create_new_interpolator': False,    # Use existing interpolators
            'snr_method': "interpolation_aligned_spins",      # Standard Numba aligned spins backend
        })
        
        gwsnr_numba = GWSNR(**config_numba)
        snr_numba = gwsnr_numba.optimal_snr(gw_param_dict=param_dict)
        
        # Cross-validate: JAX and Numba should produce reasonably consistent results
        # (Allow some tolerance due to different interpolation implementations)
        np.testing.assert_allclose(
            snr_result["snr_net"],    # JAX network SNR
            snr_numba["snr_net"],  # Numba network SNR
            rtol=0.1,                      # Allow 10% relative difference
            err_msg="JAX and Numba backends should produce similar SNR values"
        )

    def test_jax_no_spins_interpolation(self):
        """
        Tests
        -----
        - JAX no spins interpolation: "interpolation_no_spins_jax" backend
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """
        # Configure GWSNR for JAX no spins interpolation
        config = JAX_CONFIG.copy()
        config.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
            'snr_method': "interpolation_no_spins_jax"  # Use no-spins interpolation method
        })
        
        # Initialize GWSNR with JAX aligned spins configuration
        gwsnr = GWSNR(**config)
        
        # Generate parameters for BBH events with zero spins
        nsamples = 20
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',
            spin_zero=True,
            spin_precession=False
        )
        
        # Calculate SNR using JAX no-spins backend
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        
        # Validate output
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)