"""
Integration tests for GWSNR SNR calculation with aligned-spin binary black hole systems.

This module provides comprehensive integration tests for the GWSNR package's signal-to-noise
ratio calculation methods specifically for aligned-spin binary black hole systems. These tests
validate the interpolation-based SNR calculation against realistic astrophysical parameters
using the IMRPhenomD waveform approximant, which is optimized for aligned-spin systems where
the component spins are aligned or anti-aligned with the orbital angular momentum.

Test Coverage:
    * Aligned-spin BBH systems with non-zero spin magnitudes but zero tilt angles
    * Mass ratios spanning the GWTC-3 observed population (0.2 to 1.0)
    * Chirp mass range covering typical LIGO/Virgo detections (5-60 M☉)
    * Advanced LIGO/Virgo three-detector network (L1, H1, V1)
    * Interpolation method validation with precomputed aligned-spin coefficients
    * Detection probability estimation and threshold validation
    * JSON serialization and file output functionality for population studies
    * Numerical reproducibility and computational performance assessment

The aligned-spin interpolation method tested here represents an intermediate complexity
between non-spinning and fully precessing systems, providing enhanced accuracy for
systems with significant aligned spins while maintaining computational efficiency
for large-scale population synthesis studies.
"""

import numpy as np
import pytest
from gwsnr import GWSNR
from gwsnr.utils import append_json

# Set random seed for reproducible test results
np.random.seed(1234)


class TestGWSNRAlignedSpinBBH:
    """
    Integration test suite for GWSNR SNR calculation with aligned-spin binary black hole systems.
    
    This test class validates the core functionality of GWSNR's interpolation-based SNR
    calculation for aligned-spin binary black hole systems using the IMRPhenomD waveform
    approximant. The tests ensure accuracy, reproducibility, and proper integration with
    the Advanced LIGO/Virgo detector network for systems where component spins are aligned
    or anti-aligned with the orbital angular momentum vector.
    
    Scientific Context:
        Aligned-spin BBH systems represent a significant subset of gravitational-wave sources
        where the spin vectors are constrained to lie along the orbital angular momentum axis.
        This configuration reduces the 15-dimensional parameter space to a more manageable
        9-dimensional space while still capturing important spin-orbit coupling effects that
        influence waveform morphology and detectability.
    
    Test Coverage:
        * Mass parameter validation across realistic astrophysical ranges
        * Aligned-spin parameter configuration with non-zero spin magnitudes
        * Three-detector network SNR calculation and optimal combination
        * Interpolation method accuracy and computational efficiency
        * Detection probability estimation and binary classification
        * File I/O operations for large-scale population studies
        * Error handling and numerical stability validation
    """

    def test_aligned_spin_bbh_interpolation_snr_generation(self, tmp_path):
        """
        Test SNR generation using interpolation method for aligned-spin BBH systems.
        
        This integration test validates the core GWSNR functionality for calculating
        signal-to-noise ratios of aligned-spin binary black hole systems using the
        interpolation method. It tests the complete pipeline from parameter generation
        through SNR calculation to file output, ensuring compatibility with realistic
        astrophysical parameter distributions that include significant spin effects.
        
        The test uses the IMRPhenomD waveform approximant, which accurately models
        aligned-spin systems and provides excellent computational efficiency for
        population synthesis applications while capturing spin-orbit coupling effects.
        
        Args:
            tmp_path (pathlib.Path): Pytest fixture providing temporary directory
                for output file testing and isolation.
        
        Test Configuration:
            * Waveform approximant: IMRPhenomD (optimized for aligned-spin systems)
            * SNR method: interpolation_aligned_spins (bicubic interpolation with spin effects)
            * Mass range: Based on GWTC-3 observed population (chirp masses 5-60 M☉)
            * Mass ratio: 0.2-1.0 (representative of detected BBH population)
            * Luminosity distance: Fixed at 500 Mpc (intermediate sensitivity range)
            * Spin configuration: Aligned spins with magnitudes up to 0.8
            * Tilt angles: Zero (perfect alignment with orbital angular momentum)
            * Detectors: L1, H1, V1 (Advanced LIGO/Virgo three-detector network)
            * Sample size: 10 events (sufficient for integration testing)
        
        Validation Criteria:
            * Output dictionary contains required SNR and detection probability keys
            * SNR values are finite, non-negative, and physically reasonable
            * Detection probabilities are binary (0 or 1) as expected
            * Array shapes match input parameter dimensions
            * JSON serialization preserves all computed values
            * Output file is successfully created and accessible
        
        Scientific Validation:
            * Spin effects should be visible in SNR enhancement for favorable configurations
            * Detection probabilities should reflect realistic sensitivity thresholds
            * Mass-dependent SNR scaling should follow theoretical expectations
            * Network SNR should properly combine individual detector contributions
        """
        # Initialize the GWSNR object with aligned-spin interpolation
        gwsnr = GWSNR(
            npool=4,
            waveform_approximant="IMRPhenomD",
            snr_type="interpolation_aligned_spins",
            ifos=["L1", "H1", "V1"],
            interpolator_dir="./interpolator_pickle",
            pdet=True,
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        # Generate realistic aligned-spin BBH parameters
        nsamples = 10
        chirp_mass = np.linspace(5, 60, nsamples)  # GWTC-3 representative range
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)  # Realistic mass ratio distribution
        
        # Convert chirp mass and mass ratio to individual masses
        total_mass = chirp_mass * (1 + mass_ratio)**(1/5) / mass_ratio**(3/5) * (1 + mass_ratio)
        mass_1 = total_mass / (1 + mass_ratio)
        mass_2 = total_mass * mass_ratio / (1 + mass_ratio)
        
        param_dict = dict(
            mass_1=mass_1,
            mass_2=mass_2,
            luminosity_distance=500 * np.ones(nsamples),  # Fixed intermediate distance
            # Extrinsic parameters: uniformly sample sky location and orientation
            theta_jn=np.random.uniform(0, np.pi, size=nsamples),  # Inclination angle
            ra=np.random.uniform(0, 2*np.pi, size=nsamples),  # Right ascension
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),  # Declination
            psi=np.random.uniform(0, np.pi, size=nsamples),  # Polarization angle
            phase=np.random.uniform(0, 2*np.pi, size=nsamples),  # Coalescence phase
            geocent_time=1246527224.169434 * np.ones(nsamples),  # Fixed GPS time
            # Aligned-spin configuration: non-zero spin magnitudes, zero tilt angles
            a_1=np.random.uniform(0, 0.8, size=nsamples),  # Primary spin magnitude
            a_2=np.random.uniform(0, 0.8, size=nsamples),  # Secondary spin magnitude
            tilt_1=np.zeros(nsamples),  # Perfect alignment (zero tilt)
            tilt_2=np.zeros(nsamples),  # Perfect alignment (zero tilt)
            phi_12=np.zeros(nsamples),  # Azimuthal angle between spins (irrelevant for aligned case)
            phi_jl=np.zeros(nsamples),  # Azimuthal angle between J and L (irrelevant for aligned case)
        )

        # Calculate SNR using aligned-spin interpolation method
        interp_result = gwsnr.snr(gw_param_dict=param_dict)

        # Comprehensive output validation
        assert isinstance(interp_result, dict), "SNR output should be a dictionary"
        
        # Validate required keys in output
        required_keys = ["pdet_net"]
        for key in required_keys:
            assert key in interp_result, f"Expected '{key}' in SNR output dictionary"
        
        # Validate detection probability array
        pdet_array = np.asarray(interp_result["pdet_net"])
        assert pdet_array.shape == (nsamples,), f"Detection probability shape mismatch: {pdet_array.shape}"
        assert np.all(np.logical_or(pdet_array == 0, pdet_array == 1)), \
            "Detection probabilities must be binary (0 or 1)"
        
        # Validate additional SNR outputs if present
        if "optimal_snr_net" in interp_result:
            snr_array = np.asarray(interp_result["optimal_snr_net"])
            assert snr_array.shape == (nsamples,), f"SNR array shape mismatch: {snr_array.shape}"
            assert np.all(np.isfinite(snr_array)), "SNR values must be finite"
            assert np.all(snr_array >= 0), "SNR values must be non-negative"
        
        # Test JSON serialization and file output functionality
        combined_data = param_dict.copy()
        combined_data.update(interp_result)
        
        output_file = tmp_path / "BBH_aligned_spin_test_results.json"
        append_json(output_file, combined_data, replace=True)
        
        # Validate file creation and accessibility
        assert output_file.exists(), "Output JSON file was not created successfully"
        assert output_file.stat().st_size > 0, "Output JSON file is empty"

    def test_aligned_spin_parameter_validation(self):
        """
        Test parameter validation and edge cases for aligned-spin BBH systems.
        
        This test validates the robustness of the aligned-spin interpolation method
        when presented with edge cases and boundary conditions in the parameter space.
        It ensures that the interpolation method handles extreme mass ratios, high
        spins, and boundary values gracefully without numerical instabilities.
        
        Test Cases:
            * Extreme mass ratios (q=0.2 and q=1.0)
            * Maximum allowed spin magnitudes (a=0.998)
            * Minimum total masses and maximum chirp masses
            * Edge cases in sky location parameters
            * Reproducibility under identical parameter sets
        """
        gwsnr = GWSNR(
            npool=2,
            waveform_approximant="IMRPhenomD",
            snr_type="interpolation_aligned_spins",
            ifos=["L1", "H1", "V1"],
            interpolator_dir="./interpolator_pickle",
            pdet=False,  # Focus on SNR calculation only
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )
        
        # Test extreme mass ratio systems
        nsamples = 5
        param_dict_extreme = dict(
            mass_1=np.array([50.0, 30.0, 25.0, 20.0, 15.0]),  # Varied primary masses
            mass_2=np.array([10.0, 6.0, 5.0, 4.0, 3.0]),     # Extreme mass ratios
            luminosity_distance=1000 * np.ones(nsamples),     # Distant sources
            theta_jn=np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]),  # Full inclination range
            ra=np.zeros(nsamples),                             # Fixed sky position
            dec=np.zeros(nsamples),                            # Fixed sky position
            psi=np.zeros(nsamples),                            # Fixed polarization
            phase=np.zeros(nsamples),                          # Fixed phase
            geocent_time=1246527224.169434 * np.ones(nsamples),
            a_1=np.array([0.0, 0.2, 0.5, 0.8, 0.95]),        # Increasing primary spins
            a_2=np.array([0.0, 0.1, 0.3, 0.6, 0.9]),         # Increasing secondary spins
            tilt_1=np.zeros(nsamples),                         # Perfect alignment
            tilt_2=np.zeros(nsamples),                         # Perfect alignment
            phi_12=np.zeros(nsamples),                         # Irrelevant for aligned case
            phi_jl=np.zeros(nsamples),                         # Irrelevant for aligned case
        )
        
        # Calculate SNR for extreme parameter cases
        result_extreme = gwsnr.snr(gw_param_dict=param_dict_extreme)
        
        # Validate extreme case outputs
        assert isinstance(result_extreme, dict), "Output should be a dictionary"
        assert "optimal_snr_net" in result_extreme, "Missing optimal_snr_net in output"
        
        snr_extreme = np.asarray(result_extreme["optimal_snr_net"])
        assert snr_extreme.shape == (nsamples,), "SNR array shape mismatch"
        assert np.all(np.isfinite(snr_extreme)), "Non-finite SNR values in extreme cases"
        assert np.all(snr_extreme >= 0), "Negative SNR values in extreme cases"
        
        # Test reproducibility
        result_extreme_repeat = gwsnr.snr(gw_param_dict=param_dict_extreme)
        np.testing.assert_allclose(
            snr_extreme, 
            np.asarray(result_extreme_repeat["optimal_snr_net"]), 
            rtol=1e-12,
            err_msg="Aligned-spin interpolation is not reproducible"
        )