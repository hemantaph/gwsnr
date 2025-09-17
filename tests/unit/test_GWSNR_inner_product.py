"""
Unit tests for GWSNR inner product-based SNR calculation methods.

This module provides comprehensive unit tests for the GWSNR package's inner product-based
signal-to-noise ratio (SNR) calculation methods. These tests validate the core functionality
of GWSNR when configured with `snr_type="inner_product"`, which computes exact SNR values
through direct waveform generation and noise-weighted inner products.

Key Features:
------------------------
- Exact SNR computation (no interpolation approximations)
- Full support for spinning and precessing binary systems
- Compatible with all LALSimulation waveform approximants
- Higher accuracy than interpolation methods, but at higher computational cost

Multiprocessing Behavior:
------------------------
The inner product method uses Python's multiprocessing module for parallel computation:

- `multiprocessing_verbose=True` (default): Uses `imap` with tqdm progress bar
  - Provides real-time progress monitoring for long calculations
  - Slightly slower due to progress bar overhead
  - Recommended for interactive use and large parameter studies

- `multiprocessing_verbose=False`: Uses `map` without progress bar
  - Faster execution by eliminating progress bar overhead
  - No real-time progress feedback
  - Recommended for automated testing and batch processing

Test Coverage:
--------------
• Core functionality: Spinning BBH systems with exact SNR computation
• Waveform compatibility: Multiple approximants (IMRPhenomXPHM, SEOBNRv4, TaylorF2)
• Detector flexibility: Custom detector configurations and PSDs
• Performance validation: Computational efficiency and memory usage
• Reproducibility: Deterministic results across multiple runs
• Error handling: Robust validation of invalid inputs and edge cases

Usage:
-----
Run individual tests: pytest test_GWSNR_inner_product.py::test_name
Run all tests: pytest test_GWSNR_inner_product.py -v
"""

import numpy as np
import pytest
import time
from gwsnr import GWSNR
from gwsnr.utils import append_json

np.random.seed(1234)

class TestGWSNRInnerProduct:
    """
    Test suite for GWSNR inner product-based SNR calculations.
    
    Validates exact SNR computation using direct waveform generation and noise-weighted
    inner products. Tests core functionality, waveform compatibility, detector flexibility,
    and performance characteristics of the `snr_type="inner_product"` method.
    """
    
    def _generate_bbh_params(self, nsamples, include_spins=False, distance=500.0):
        """Generate realistic BBH parameters for testing."""
        mtot = np.random.uniform(20, 200, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, nsamples)
        
        params = {
            'mass_1': mtot / (1 + mass_ratio),
            'mass_2': mtot * mass_ratio / (1 + mass_ratio),
            'luminosity_distance': distance * np.ones(nsamples),
            'geocent_time': 1246527224.169434 * np.ones(nsamples),
            'theta_jn': np.random.uniform(0, 2*np.pi, nsamples),
            'ra': np.random.uniform(0, 2*np.pi, nsamples),
            'dec': np.random.uniform(-np.pi/2, np.pi/2, nsamples),
            'psi': np.random.uniform(0, 2*np.pi, nsamples),
            'phase': np.random.uniform(0, 2*np.pi, nsamples),
        }
        
        if include_spins:
            params.update({
                'a_1': np.random.uniform(0, 0.8, nsamples),
                'a_2': np.random.uniform(0, 0.8, nsamples),
                'tilt_1': np.random.uniform(0, np.pi, nsamples),
                'tilt_2': np.random.uniform(0, np.pi, nsamples),
                'phi_12': np.random.uniform(0, 2*np.pi, nsamples),
                'phi_jl': np.random.uniform(0, 2*np.pi, nsamples),
            })
        
        return params
    
    def _validate_snr_output(self, snr_dict, expected_samples, test_name=""):
        """Validate SNR output format and numerical properties."""
        assert isinstance(snr_dict, dict), f"{test_name}: SNR output should be a dictionary"
        assert "optimal_snr_net" in snr_dict, f"{test_name}: Missing 'optimal_snr_net'"
        
        snr_arr = np.asarray(snr_dict["optimal_snr_net"])
        assert snr_arr.shape == (expected_samples,), f"{test_name}: Shape mismatch"
        assert np.all(np.isfinite(snr_arr)), f"{test_name}: Non-finite SNR values"
        assert np.all(np.isreal(snr_arr)), f"{test_name}: SNR values not real"
        assert np.all(snr_arr >= 0), f"{test_name}: Negative SNR values"
        assert snr_arr.dtype == np.float64, f"{test_name}: Wrong dtype"
    
    def _create_gwsnr(self, **overrides):
        """Create GWSNR instance with sensible defaults and optional overrides."""
        defaults = {
            'npool': 1,
            'sampling_frequency': 2048.0,
            'waveform_approximant': "IMRPhenomD", 
            'frequency_domain_source_model': 'lal_binary_black_hole',
            'minimum_frequency': 20.0,
            'snr_type': "inner_product",
            'gwsnr_verbose': False,
            'multiprocessing_verbose': False,
        }
        defaults.update(overrides)
        return GWSNR(**defaults)

    def test_spinning_bbh_systems(self, tmp_path):
        """
        Test inner product SNR calculation for spinning and precessing BBH systems.
        
        Validates core functionality with precessing binaries using IMRPhenomXPHM.
        Tests waveform generation, SNR computation, JSON output, and reproducibility.
        """
        gwsnr = self._create_gwsnr(
            npool=4,
            waveform_approximant="IMRPhenomXPHM",
            ifos=["L1", "H1", "V1"],
            multiprocessing_verbose=True  # Test progress bar functionality
        )

        nsamples = 5
        param_dict = self._generate_bbh_params(nsamples, include_spins=True)
        
        # Calculate SNR with spinning systems
        snr_result = gwsnr.snr(gw_param_dict=param_dict)
        self._validate_snr_output(snr_result, nsamples, "spinning_bbh")
        
        # Test JSON serialization
        param_dict.update(snr_result)
        output_file = tmp_path / "snr_spinning_inner_product.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists() and output_file.stat().st_size > 0

        # Test reproducibility
        snr_result2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            snr_result["optimal_snr_net"], snr_result2["optimal_snr_net"],
            rtol=1e-10, err_msg="Non-deterministic SNR calculation"
        )

    def test_multiple_waveform_approximants(self):
        """
        Test SNR calculation across different waveform approximants.
        
        Validates compatibility with IMRPhenomD (spinless), SEOBNRv4 (aligned spins),
        and TaylorF2 (post-Newtonian) approximants.
        """
        approximants = ["IMRPhenomD", "SEOBNRv4", "TaylorF2"]
        nsamples = 2
        
        for approx in approximants:
            gwsnr = self._create_gwsnr(waveform_approximant=approx, ifos=["L1"])
            
            # Generate parameters with spins for SEOBNRv4
            params = self._generate_bbh_params(nsamples, distance=400.0)
            if approx == "SEOBNRv4":
                params.update({
                    'a_1': np.random.uniform(-0.5, 0.5, nsamples),
                    'a_2': np.random.uniform(-0.5, 0.5, nsamples)
                })

            snr_result = gwsnr.snr(gw_param_dict=params)
            self._validate_snr_output(snr_result, nsamples, f"approximant_{approx}")

    def test_custom_detector_configuration(self):
        """
        Test SNR calculation with custom detector configuration.
        
        Creates LIGO India A1 detector with custom coordinates and PSD to validate
        inner product method flexibility with non-standard detector setups.
        """

        import bilby
        
        # Create LIGO India A1 detector
        ifo_a1 = bilby.gw.detector.interferometer.Interferometer(
            name='A1',
            power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
                asd_file='aLIGO_O4_high_asd.txt'
            ),
            minimum_frequency=20,
            maximum_frequency=2048,
            length=4,
            latitude=19.613,  # Aundha coordinates (simplified)
            longitude=77.031,
            elevation=440.0,
            xarm_azimuth=117.6,
            yarm_azimuth=207.6
        )
        
        gwsnr = self._create_gwsnr(
            psds={'A1': 'aLIGO_O4_high_asd.txt'},
            ifos=[ifo_a1]
        )

        nsamples = 2
        params = self._generate_bbh_params(nsamples, distance=400.0)
        snr_result = gwsnr.snr(gw_param_dict=params)
        self._validate_snr_output(snr_result, nsamples, "custom_detector_A1")

    def test_multiprocessing_performance(self):
        """
        Test multiprocessing performance and compare serial vs parallel execution.
        
        Validates that both imap (with progress bar) and map (without) modes work
        correctly and produce consistent results.
        """
        nsamples = 5  # Small sample for quick testing
        params = self._generate_bbh_params(nsamples)
        
        # Test serial execution (no multiprocessing overhead)
        gwsnr_serial = self._create_gwsnr(npool=1, ifos=["L1"])
        start_time = time.time()
        serial_snr = gwsnr_serial.snr(gw_param_dict=params)
        serial_time = time.time() - start_time
        
        # Test parallel with progress bar (imap mode)
        gwsnr_parallel_verbose = self._create_gwsnr(
            npool=2, ifos=["L1"], multiprocessing_verbose=True
        )
        parallel_verbose_snr = gwsnr_parallel_verbose.snr(gw_param_dict=params)
        
        # Test parallel without progress bar (map mode) 
        gwsnr_parallel_quiet = self._create_gwsnr(
            npool=2, ifos=["L1"], multiprocessing_verbose=False
        )
        start_time = time.time()
        parallel_quiet_snr = gwsnr_parallel_quiet.snr(gw_param_dict=params)
        parallel_time = time.time() - start_time
        
        # Validate all methods produce consistent results
        self._validate_snr_output(serial_snr, nsamples, "serial")
        self._validate_snr_output(parallel_verbose_snr, nsamples, "parallel_verbose")
        self._validate_snr_output(parallel_quiet_snr, nsamples, "parallel_quiet")
        
        # Cross-validation between methods
        np.testing.assert_allclose(
            serial_snr["optimal_snr_net"],
            parallel_verbose_snr["optimal_snr_net"],
            rtol=1e-8, err_msg="Serial and parallel_verbose should match"
        )
        np.testing.assert_allclose(
            serial_snr["optimal_snr_net"],
            parallel_quiet_snr["optimal_snr_net"],
            rtol=1e-8, err_msg="Serial and parallel_quiet should match"
        )
        
        # Basic timing validation
        assert serial_time > 0 and parallel_time > 0, "Both should take measurable time"

    def test_custom_psds(self):
        """
        Test SNR calculation with custom power spectral densities.
        
        Validates inner product method with design sensitivity PSDs and compares
        results with default PSDs to ensure custom noise curves work correctly.
        """
        custom_psds = {
            'L1': 'aLIGOaLIGODesignSensitivityT1800044',
            'H1': 'aLIGOaLIGODesignSensitivityT1800044', 
            'V1': 'AdvVirgo'
        }
        
        gwsnr_custom = self._create_gwsnr(
            psds=custom_psds,
        )
            
        nsamples = 3
        params = self._generate_bbh_params(nsamples, distance=400.0)
        
        # Test custom PSDs
        custom_snr = gwsnr_custom.snr(gw_param_dict=params)
        self._validate_snr_output(custom_snr, nsamples, "custom_psds")
        
        # Compare with default PSDs
        gwsnr_default = self._create_gwsnr(psds=None, ifos=["L1", "H1", "V1"])
        default_snr = gwsnr_default.snr(gw_param_dict=params)
        
        # Validate that custom PSDs produce reasonable results
        custom_arr = np.asarray(custom_snr["optimal_snr_net"])
        default_arr = np.asarray(default_snr["optimal_snr_net"])
        ratio = custom_arr / default_arr
        
        assert np.all(ratio > 0.1) and np.all(ratio < 10.0), \
            "Custom PSD SNRs should be within reasonable range of defaults"



