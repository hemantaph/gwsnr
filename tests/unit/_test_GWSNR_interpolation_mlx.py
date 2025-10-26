"""
Unit Tests for GWSNR MLX-Accelerated Interpolation Backend. This test can only run on Apple Silicon (ARM64 macOS) with MLX installed.

IMPORTANT: These tests only run on Apple Silicon (ARM64 macOS) with MLX package installed.
Tests will be automatically skipped on other platforms or when MLX is not available.

Requirements:
-------------
- pip install gwsnr mlx
- pip install pytest

Test Coverage:
--------------
- MLX aligned spins interpolation: "interpolation_aligned_spins_mlx" backend, using IMRPhenomD waveform model
- MLX no spins interpolation: "interpolation_no_spins_mlx" backend, using IMRPhenomD waveform model
- Output validation: dictionary structure, data types, shapes, numerical properties
- Reproducibility testing: Deterministic output verification after JIT compilation
- Cross-validation: Comparison with standard Numba backend results

Usage:
-----
First change the name of this file to test_GWSNR_interpolation_mlx.py

pytest tests/unit/test_GWSNR_interpolation_mlx.py -v -s
pytest tests/unit/test_GWSNR_interpolation_mlx.py::TestGWSNRInterpolationMLX::test_name -v -s
"""

import numpy as np
import platform
import pytest
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

np.random.seed(1234)

# Check if running on Apple Silicon (ARM64 architecture on macOS)
is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

# Check if MLX is available (only on Apple Silicon)
mlx_available = False
if is_apple_silicon:
    try:
        import mlx
        mlx_available = True
    except ImportError:
        mlx_available = False

# Skip all tests if not on Apple Silicon or MLX not available
skip_mlx_tests = not (is_apple_silicon and mlx_available)

# MLX-specific GWSNR configuration dictionary for all tests
# Optimized for MLX backend performance with reduced resolution for faster testing
MLX_CONFIG = {
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
    
    # MLX-specific SNR calculation settings
    'snr_method': "interpolation_aligned_spins_mlx", # Default to MLX aligned spins backend
    'interpolator_dir': "./interpolator_pickle", # Directory for saved interpolators
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

class TestGWSNRInterpolationMLX(CommonTestUtils):
    """Test suite for GWSNR MLX-accelerated interpolation backends."""

    @pytest.mark.skipif(skip_mlx_tests, reason="MLX tests require Apple Silicon (ARM64 macOS) with MLX package installed")
    def test_mlx_aligned_spins_interpolation(self):
        """
        Tests
        -----
        - MLX aligned spins interpolation: "interpolation_aligned_spins_mlx" backend
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Reproducibility testing: Deterministic output verification after JIT compilation
        - Cross-validation: Comparison with standard Numba backend results
        """
        config = MLX_CONFIG.copy()
        config.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
        })
        
        # Initialize GWSNR with MLX configuration
        gwsnr = GWSNR(**config)
        
        # Generate parameters for BBH events with aligned spins
        nsamples = 20
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',
            spin_zero=False,
            spin_precession=False
        )
        
        # Calculate SNR using MLX backend  
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        
        # Validate MLX output structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)
        
        # Test MLX reproducibility (JIT compilation should be deterministic)
        snr_result2 = gwsnr.optimal_snr(gw_param_dict=param_dict)  # Second call uses compiled function
        np.testing.assert_allclose(
            snr_result["snr_net"],   # Network SNR from first calculation
            snr_result2["snr_net"],  # Network SNR from second calculation (JIT compiled)
            rtol=1e-10,                      # Very tight tolerance for MLX determinism
            err_msg="MLX backend should be deterministic after JIT compilation"
        )
        
        ################################################
        # Cross-validation with standard Numba backend #
        ################################################
        # Standard Numba backend for comparison
        config_numba = MLX_CONFIG.copy()  # Use same base config
        config_numba.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
            'create_new_interpolator': False,    # Use existing interpolators
            'snr_method': "interpolation_aligned_spins",      # Standard Numba aligned spins backend
        })
        
        gwsnr_numba = GWSNR(**config_numba)
        snr_numba = gwsnr_numba.optimal_snr(gw_param_dict=param_dict)
        
        # Cross-validate: MLX and Numba should produce reasonably consistent results
        # (Allow some tolerance due to different interpolation implementations)
        np.testing.assert_allclose(
            snr_result["snr_net"],    # MLX network SNR
            snr_numba["snr_net"],  # Numba network SNR
            rtol=0.1,                      # Allow 10% relative difference
            err_msg="MLX and Numba backends should produce similar SNR values"
        )

    @pytest.mark.skipif(skip_mlx_tests, reason="MLX tests require Apple Silicon (ARM64 macOS) with MLX package installed")
    def test_mlx_no_spins_interpolation(self):
        """
        Tests
        -----
        - MLX no spins interpolation: "interpolation_no_spins_mlx" backend
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """
        # Configure GWSNR for MLX no spins interpolation
        config = MLX_CONFIG.copy()
        config.update({
            'gwsnr_verbose': False,              # Reduce log output for cleaner tests
            'snr_method': "interpolation_no_spins_mlx"  # Use no-spins interpolation method
        })
        
        # Initialize GWSNR with MLX aligned spins configuration
        gwsnr = GWSNR(**config)
        
        # Generate parameters for BBH events with zero spins
        nsamples = 20
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',
            spin_zero=True,
            spin_precession=False
        )
        
        # Calculate SNR using MLX no-spins backend
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        
        # Validate output
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)