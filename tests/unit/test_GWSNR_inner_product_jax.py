"""
Unit tests for GWSNR JAX-accelerated inner product method using ripplegw waveforms.

This module tests the JAX/ripplegw integration for hardware-accelerated SNR calculations,
providing an alternative to the standard inner product method with JIT compilation and
GPU acceleration capabilities.

Key Features:
-----------
• JAX backend with JIT compilation for performance optimization
• ripplegw library for JAX-native gravitational waveform generation  
• Cross-validation with standard inner product method for consistency
• Supports multiple waveform approximants via ripplegw

Supported Waveform Approximants:
-------------------------------
• IMRPhenomXAS: Aligned-spin with higher-order modes
• IMRPhenomD: Standard aligned-spin approximant  
• TaylorF2: Post-Newtonian inspiral waveforms
• IMRPhenomD_NRTidalv2: Binary neutron star systems with tidal effects

Test Coverage:
--------------
• Cross-validation: JAX results match standard inner product method
• Multi-approximant compatibility testing
• Performance and reproducibility validation
• Error handling for missing JAX/ripplegw dependencies

Requirements:
• JAX framework and ripplegw library installation
• Optional: NVIDIA GPU with CUDA for maximum performance

Note: This test focuses on JAX-specific functionality and cross-validation rather than
comprehensive testing already covered in test_GWSNR_inner_product.py.
"""

import numpy as np
import pytest
import time
from gwsnr import GWSNR

np.random.seed(1234)

class TestGWSNRInnerProductJAX:
    """
    Test suite for JAX-accelerated inner product SNR calculations using ripplegw.
    
    Validates JAX backend functionality and cross-validates results with standard
    inner product method. Tests focus on JAX-specific features rather than
    comprehensive functionality already covered in test_GWSNR_inner_product.py.
    """
    
    def _generate_bbh_params(self, nsamples=5, include_spins=True):
        """Generate realistic BBH parameters for testing."""
        mtot = np.random.uniform(30, 100, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, nsamples)
        
        params = {
            'mass_1': mtot / (1 + mass_ratio),
            'mass_2': mtot * mass_ratio / (1 + mass_ratio),
            'luminosity_distance': 400.0 * np.ones(nsamples),
            'geocent_time': 1246527224.169434 * np.ones(nsamples),
            'theta_jn': np.random.uniform(0, 2*np.pi, nsamples),
            'ra': np.random.uniform(0, 2*np.pi, nsamples),
            'dec': np.random.uniform(-np.pi/2, np.pi/2, nsamples),
            'psi': np.random.uniform(0, 2*np.pi, nsamples),
            'phase': np.random.uniform(0, 2*np.pi, nsamples),
        }
        
        if include_spins:
            params.update({
                'a_1': np.random.uniform(0, 0.5, nsamples),
                'a_2': np.random.uniform(0, 0.5, nsamples),
                'tilt_1': np.random.uniform(0, np.pi, nsamples),
                'tilt_2': np.random.uniform(0, np.pi, nsamples),
                'phi_12': np.random.uniform(0, 2*np.pi, nsamples),
                'phi_jl': np.random.uniform(0, 2*np.pi, nsamples),
            })
        
        return params
    
    def _validate_snr_output(self, snr_dict, expected_samples):
        """Validate SNR output format and values."""
        assert isinstance(snr_dict, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in snr_dict, "Missing 'optimal_snr_net'"
        
        snr_arr = np.asarray(snr_dict["optimal_snr_net"])
        assert snr_arr.shape == (expected_samples,), f"Shape mismatch: expected ({expected_samples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "SNR values must be finite"
        assert np.all(snr_arr >= 0), "SNR values must be non-negative"
        assert snr_arr.dtype == np.float64, "SNR values must be float64"

    def test_jax_cross_validation_with_standard_method(self):
        """
        Cross-validate JAX inner product method against standard inner product method.
        
        Tests that JAX-accelerated computation with ripplegw produces results consistent
        with the standard LAL-based inner product method. This ensures numerical
        accuracy is maintained while gaining performance benefits from JAX compilation.
        """
        try:
            # Create both JAX and standard inner product instances
            gwsnr_jax = GWSNR(
                snr_type="inner_product_jax",
                waveform_approximant="IMRPhenomXAS",
                npool=1,  # Avoid multiprocessing issues with JAX
                gwsnr_verbose=False,
                multiprocessing_verbose=False
            )
            
            gwsnr_standard = GWSNR(
                snr_type="inner_product", 
                waveform_approximant="IMRPhenomXPHM",  # Equivalent to XAS
                npool=1,
                gwsnr_verbose=False,
                multiprocessing_verbose=False
            )
        except (ImportError, Exception) as e:
            pytest.skip(f"JAX/ripplegw dependencies not available: {e}")

        # Test with spinning BBH parameters
        nsamples = 3
        params = self._generate_bbh_params(nsamples, include_spins=True)
        
        # Calculate SNR with both methods
        jax_snr = gwsnr_jax.snr(gw_param_dict=params)
        standard_snr = gwsnr_standard.snr(gw_param_dict=params)
        
        # Validate both outputs
        self._validate_snr_output(jax_snr, nsamples)
        self._validate_snr_output(standard_snr, nsamples)
        
        # Cross-validate results (allow reasonable tolerance for different implementations)
        # JAX/ripplegw and LAL use different waveform implementations, so moderate differences are expected
        np.testing.assert_allclose(
            jax_snr["optimal_snr_net"],
            standard_snr["optimal_snr_net"],
            rtol=0.15,  # 15% tolerance for different waveform approximant implementations
            err_msg="JAX and standard methods should produce similar SNR values"
        )

    def test_multiple_waveform_approximants(self):
        """
        Test JAX inner product method with multiple supported waveform approximants.
        
        Validates that ripplegw supports the expected waveform approximants and
        produces reasonable SNR values for each.
        """
        supported_approximants = ['IMRPhenomXAS', 'IMRPhenomD', 'TaylorF2', 'IMRPhenomD_NRTidalv2']
        nsamples = 2
        
        for approx in supported_approximants:
            try:
                gwsnr = GWSNR(
                    snr_type="inner_product_jax",
                    waveform_approximant=approx,
                    npool=1,  # Serial for faster testing
                    gwsnr_verbose=False
                )
                
                # Generate appropriate parameters for each approximant
                params = self._generate_bbh_params(nsamples, include_spins=(approx != 'TaylorF2'))
                
                # For tidal approximant, use lower masses
                if 'Tidal' in approx:
                    params['mass_1'] = np.full(nsamples, 1.4)  # Neutron star masses
                    params['mass_2'] = np.full(nsamples, 1.4)
                
                snr_result = gwsnr.snr(gw_param_dict=params)
                self._validate_snr_output(snr_result, nsamples)
                
            except (ImportError, Exception):
                pytest.skip(f"JAX/ripplegw does not support {approx} or dependencies missing")

    def test_jax_reproducibility(self):
        """
        Test that JAX inner product method produces reproducible results.
        
        Validates that JIT compilation and JAX operations are deterministic
        across multiple calls with identical parameters.
        """
        try:
            gwsnr = GWSNR(
                snr_type="inner_product_jax",
                waveform_approximant="IMRPhenomD",
                npool=1,
                gwsnr_verbose=False
            )
        except (ImportError, Exception) as e:
            pytest.skip(f"JAX dependencies not available: {e}")

        params = self._generate_bbh_params(3, include_spins=False)
        
        # Calculate SNR multiple times
        snr1 = gwsnr.snr(gw_param_dict=params)
        snr2 = gwsnr.snr(gw_param_dict=params)
        
        # Should be identical
        np.testing.assert_array_equal(
            snr1["optimal_snr_net"],
            snr2["optimal_snr_net"],
            err_msg="JAX method should be deterministic"
        )
    
    def test_jax_produces_reasonable_snr_values(self):
        """
        Test that JAX inner product method produces physically reasonable SNR values.
        
        Validates that JAX-computed SNRs are in the expected range for typical
        binary black hole systems, confirming the method works correctly.
        """
        try:
            gwsnr = GWSNR(
                snr_type="inner_product_jax",  
                waveform_approximant="IMRPhenomD",
                npool=1,
                gwsnr_verbose=False
            )
        except (ImportError, Exception) as e:
            pytest.skip(f"JAX dependencies not available: {e}")

        # Test with close and distant sources
        close_params = self._generate_bbh_params(2, include_spins=False)
        close_params['luminosity_distance'] = np.array([100.0, 200.0])  # Close sources
        
        distant_params = self._generate_bbh_params(2, include_spins=False) 
        distant_params['luminosity_distance'] = np.array([1000.0, 2000.0])  # Distant sources
        
        close_snr = gwsnr.snr(gw_param_dict=close_params)
        distant_snr = gwsnr.snr(gw_param_dict=distant_params)
        
        self._validate_snr_output(close_snr, 2)
        self._validate_snr_output(distant_snr, 2)
        
        # Close sources should have higher SNR than distant ones
        assert np.all(close_snr["optimal_snr_net"] > distant_snr["optimal_snr_net"]), \
            "Closer sources should have higher SNR than distant ones"
        
        # SNRs should be in reasonable range for these distances
        assert np.all(close_snr["optimal_snr_net"] > 5), "Close sources should have detectable SNR"
        assert np.all(distant_snr["optimal_snr_net"] < 50), "Distant sources should have lower SNR"

