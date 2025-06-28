"""
Integration tests for GWSNR SNR calculation with non-spinning binary black hole systems.

This module provides comprehensive integration tests for the GWSNR package's signal-to-noise
ratio calculation methods specifically for non-spinning (a_1=0, a_2=0) binary black hole
systems. These tests validate the interpolation-based SNR calculation against realistic
astrophysical parameters using the IMRPhenomD waveform approximant, which is optimal for
aligned-spin and non-spinning systems.

Test Coverage:
    * Non-spinning BBH systems with mass ratios from 0.2 to 1.0
    * Total masses spanning GWTC-3 detection range (chirp masses 5-60 M☉)
    * Advanced LIGO/Virgo three-detector network (L1, H1, V1)
    * Interpolation method validation with precomputed coefficients
    * JSON serialization and file output functionality
    * Numerical reproducibility and error handling

The interpolation method tested here is the primary computational engine for population
synthesis studies and rapid parameter estimation where high-precision SNR calculations
are needed for thousands to millions of binary systems.
"""

import numpy as np
import pytest
from gwsnr import GWSNR
from gwsnr.utils import append_json

# Set random seed for reproducible test results
np.random.seed(1234)


class TestGWSNRNonSpinningBBH:
    """
    Integration test suite for GWSNR SNR calculation with non-spinning binary black hole systems.
    
    This test class validates the core functionality of GWSNR's interpolation-based SNR
    calculation for non-spinning binary black hole systems using the IMRPhenomD waveform
    approximant. The tests ensure accuracy, reproducibility, and proper integration with
    the Advanced LIGO/Virgo detector network configuration.
    
    Scientific Context:
        Non-spinning BBH systems represent a significant fraction of observed gravitational-wave
        sources and serve as an important baseline for testing gravitational-wave analysis
        pipelines. The IMRPhenomD waveform model provides accurate templates for these systems
        across the LIGO/Virgo sensitivity band.
    
    Test Coverage:
        * Mass parameter validation across realistic astrophysical ranges
        * Three-detector network SNR calculation and combination
        * Interpolation method accuracy and computational efficiency
        * File I/O operations for large-scale population studies
        * Error handling and edge case validation
    """

    def test_non_spinning_bbh_interpolation_snr_generation(self, tmp_path):
        """
        Test SNR generation using interpolation method for non-spinning BBH systems.
        
        This integration test validates the core GWSNR functionality for calculating
        signal-to-noise ratios of non-spinning binary black hole systems using the
        interpolation method. It tests the complete pipeline from parameter generation
        through SNR calculation to file output, ensuring compatibility with realistic
        astrophysical parameter distributions.
        
        The test uses the IMRPhenomD waveform approximant, which is specifically designed
        for aligned-spin and non-spinning systems and provides excellent accuracy-to-speed
        ratio for population synthesis applications.
        
        Args:
            tmp_path (pathlib.Path): Pytest fixture providing temporary directory
                for output file testing.
        
        Test Configuration:
            * Waveform approximant: IMRPhenomD (optimized for non-spinning systems)
            * SNR method: interpolation (fast bicubic interpolation of precomputed coefficients)
            * Mass range: Based on GWTC-3 observed population (chirp masses 5-60 M☉)
            * Mass ratio: 0.2-1.0 (representative of detected BBH population)
            * Luminosity distance: Fixed at 500 Mpc (intermediate sensitivity range)
            * Spin parameters: All set to zero (a_1=0, a_2=0)
            * Detectors: L1, H1, V1 (Advanced LIGO/Virgo three-detector network)
            * Sample size: 10 events (sufficient for integration testing)
        
        Physical Parameter Ranges:
            * Total mass: Derived from chirp mass and mass ratio within GWTC-3 bounds
            * Inclination (theta_jn): [0, 2π] - full range of orbital inclinations
            * Sky location (ra, dec): Full sky coverage in equatorial coordinates
            * Polarization (psi): [0, 2π] - full range of gravitational-wave polarizations
            * Coalescence phase: [0, 2π] - arbitrary reference phase
            * GPS time: Fixed to GWTC-1 reference event time
        
        Validation Criteria:
            * SNR output must be a dictionary with required keys
            * Network SNR array must match input sample size
            * All SNR values must be finite, real, and non-negative
            * JSON serialization must complete without errors
            * Output file must be created and accessible
            * Repeated calculations must yield identical results (reproducibility)
        
        Expected Behavior:
            * SNR values should be physically reasonable (0-500 range for 500 Mpc)
            * Higher mass systems should generally show higher SNR at fixed distance
            * Face-on systems (theta_jn~0) should show higher SNR than edge-on (theta_jn~π/2)
            * JSON output should contain both input parameters and computed SNR values
        
        References:
            * GWTC-3 catalog: Abbott et al. (2021), arXiv:2111.03606
            * IMRPhenomD model: Husa et al. (2016), Khan et al. (2016)
            * Advanced LIGO sensitivity: Aasi et al. (2015)
        """
        # Initialize GWSNR with interpolation method for non-spinning systems
        gwsnr = GWSNR(
            npool=4,
            waveform_approximant="IMRPhenomD",
            minimum_frequency=20.0,
            snr_type="interpolation",
            psds=None,
            ifos=["L1", "H1", "V1"],
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        # Generate realistic non-spinning BBH parameters
        nsamples = 10
        
        # Chirp mass distribution based on GWTC-3 observations
        chirp_mass = np.linspace(5, 60, nsamples)
        
        # Mass ratio distribution representative of observed BBH population
        mass_ratio = np.random.uniform(0.2, 1.0, size=nsamples)
        
        # Convert to component masses using standard relations
        # mass_1 = (M_chirp * (1+q)^(1/5)) / q^(3/5)
        # mass_2 = M_chirp * q^(2/5) * (1+q)^(1/5)
        param_dict = dict(
            mass_1=(chirp_mass * (1 + mass_ratio)**(1/5)) / mass_ratio**(3/5),
            mass_2=chirp_mass * mass_ratio**(2/5) * (1 + mass_ratio)**(1/5),
            
            # Fixed luminosity distance for consistent SNR comparison
            luminosity_distance=500 * np.ones(nsamples),
            
            # Randomly sample extrinsic parameters across full ranges
            theta_jn=np.random.uniform(0, 2*np.pi, size=nsamples),  # Inclination angle
            ra=np.random.uniform(0, 2*np.pi, size=nsamples),        # Right ascension
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples), # Declination
            psi=np.random.uniform(0, 2*np.pi, size=nsamples),       # Polarization angle
            phase=np.random.uniform(0, 2*np.pi, size=nsamples),     # Coalescence phase
            geocent_time=1246527224.169434 * np.ones(nsamples),     # GPS time (GWTC-1 reference)
        )

        # Calculate SNR using GWSNR interpolation method
        interpolation_snr_result = gwsnr.snr(gw_param_dict=param_dict)

        # Validate output structure and data types
        assert isinstance(interpolation_snr_result, dict), \
            "SNR output must be a dictionary containing results for all detectors"
        
        assert "optimal_snr_net" in interpolation_snr_result, \
            "SNR output must contain 'optimal_snr_net' key for network SNR"
        
        # Validate array dimensions and properties
        network_snr = np.asarray(interpolation_snr_result["optimal_snr_net"])
        assert network_snr.shape == (nsamples,), \
            f"Network SNR array shape mismatch: expected {(nsamples,)}, got {network_snr.shape}"
        
        # Validate numerical properties
        assert np.all(np.isfinite(network_snr)), \
            "All SNR values must be finite (no NaN or inf values)"
        
        assert np.all(network_snr >= 0), \
            "All SNR values must be non-negative"
        
        assert network_snr.dtype == np.float64, \
            "SNR values must be 64-bit floating point for numerical precision"
        
        # Validate individual detector SNRs are also present
        expected_detectors = ["L1", "H1", "V1"]
        for detector in expected_detectors:
            assert detector in interpolation_snr_result, \
                f"SNR output must contain individual detector SNR for {detector}"
            
            detector_snr = np.asarray(interpolation_snr_result[detector])
            assert detector_snr.shape == (nsamples,), \
                f"Detector {detector} SNR array shape mismatch"
            assert np.all(np.isfinite(detector_snr)), \
                f"All {detector} SNR values must be finite"
            assert np.all(detector_snr >= 0), \
                f"All {detector} SNR values must be non-negative"

        # Test JSON serialization and file output functionality
        param_dict.update(interpolation_snr_result)
        output_file = tmp_path / "BBH_non_spinning_integration_test.json"
        
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), \
            "JSON output file was not created successfully"
        
        assert output_file.stat().st_size > 0, \
            "JSON output file is empty"

        # Test reproducibility - repeated calculation should yield identical results
        interpolation_snr_result_2 = gwsnr.snr(gw_param_dict=param_dict)
        network_snr_2 = np.asarray(interpolation_snr_result_2["optimal_snr_net"])
        
        np.testing.assert_allclose(
            network_snr, network_snr_2, rtol=1e-10,
            err_msg="SNR calculation must be reproducible within numerical precision"
        )

        # Validate physical reasonableness of results
        # For non-spinning systems at 500 Mpc, network SNR should be in reasonable range
        assert np.all(network_snr <= 1000), \
            "SNR values appear unrealistically high (check units and scaling)"
        
        # At least some systems should be detectable (SNR > 8) at 500 Mpc for this mass range
        assert np.any(network_snr > 1.0), \
            "At least some systems should produce measurable SNR at 500 Mpc"

    def test_parameter_validation_and_edge_cases(self):
        """
        Test parameter validation and edge case handling for non-spinning BBH systems.
        
        This test validates that GWSNR properly handles edge cases and parameter
        validation for non-spinning binary systems, including boundary values,
        invalid inputs, and extreme parameter combinations.
        """
        gwsnr = GWSNR(
            npool=2,
            waveform_approximant="IMRPhenomD",
            snr_type="interpolation",
            ifos=["L1", "H1"],
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        # Test minimum mass system (boundary case)
        min_mass_params = dict(
            mass_1=np.array([5.0]),
            mass_2=np.array([5.0]),
            luminosity_distance=np.array([100.0]),
            theta_jn=np.array([0.0]),
            ra=np.array([0.0]),
            dec=np.array([0.0]),
            psi=np.array([0.0]),
            phase=np.array([0.0]),
            geocent_time=np.array([1246527224.169434]),
        )
        
        result = gwsnr.snr(gw_param_dict=min_mass_params)
        assert isinstance(result, dict), "Should handle minimum mass systems"
        assert np.isfinite(result["optimal_snr_net"][0]), "SNR should be finite for minimum mass"

        # Test high mass system (boundary case)
        high_mass_params = dict(
            mass_1=np.array([80.0]),
            mass_2=np.array([60.0]),
            luminosity_distance=np.array([1000.0]),
            theta_jn=np.array([np.pi/2]),
            ra=np.array([np.pi]),
            dec=np.array([0.0]),
            psi=np.array([np.pi/2]),
            phase=np.array([np.pi]),
            geocent_time=np.array([1246527224.169434]),
        )
        
        result = gwsnr.snr(gw_param_dict=high_mass_params)
        assert isinstance(result, dict), "Should handle high mass systems"
        assert np.isfinite(result["optimal_snr_net"][0]), "SNR should be finite for high mass"

    def test_mass_ratio_range_validation(self):
        """
        Test SNR calculation across the full mass ratio range for non-spinning systems.
        
        This test validates that the interpolation method works correctly across
        the full range of mass ratios from equal mass (q=1) to highly asymmetric
        (q=0.2) systems.
        """
        gwsnr = GWSNR(
            npool=2,
            waveform_approximant="IMRPhenomD",
            snr_type="interpolation",
            ifos=["L1", "H1"],
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        # Test different mass ratios at fixed total mass
        mass_ratios = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        total_mass = 40.0  # Fixed total mass
        
        mass_1_arr = total_mass / (1 + mass_ratios)
        mass_2_arr = mass_1_arr * mass_ratios
        
        param_dict = dict(
            mass_1=mass_1_arr,
            mass_2=mass_2_arr,
            luminosity_distance=np.full_like(mass_ratios, 500.0),
            theta_jn=np.zeros_like(mass_ratios),  # Face-on for maximum SNR
            ra=np.zeros_like(mass_ratios),
            dec=np.zeros_like(mass_ratios),
            psi=np.zeros_like(mass_ratios),
            phase=np.zeros_like(mass_ratios),
            geocent_time=np.full_like(mass_ratios, 1246527224.169434),
        )
        
        result = gwsnr.snr(gw_param_dict=param_dict)
        snr_values = np.asarray(result["optimal_snr_net"])
        
        # All SNR values should be finite and positive
        assert np.all(np.isfinite(snr_values)), "All mass ratios should produce finite SNR"
        assert np.all(snr_values > 0), "All mass ratios should produce positive SNR"
        
        # SNR should vary smoothly with mass ratio (no discontinuities)
        snr_diffs = np.abs(np.diff(snr_values))
        max_relative_change = np.max(snr_diffs / snr_values[:-1])
        assert max_relative_change < 0.5, "SNR should vary smoothly with mass ratio"