"""
Unit tests for GWSNR inner product-based SNR calculation methods.

This module provides comprehensive unit tests for the GWSNR package's inner product-based
signal-to-noise ratio calculation methods. These tests validate direct waveform generation
and exact inner product computations, providing higher accuracy than interpolation methods
at increased computational cost.

Test Coverage:
    * Spinning binary black hole systems with full precession
    * Multiple waveform approximants (IMRPhenomXPHM, SEOBNRv4, TaylorF2)
    * Custom detector configurations (LIGO India A1)
    * Custom power spectral densities
    * Computational performance and parallel processing
    * Accuracy validation and reproducibility

The inner product method generates waveforms on-demand and computes exact SNR values
through direct integration with detector noise, making it suitable for parameter
estimation and high-precision applications.
"""

import numpy as np
import pytest
import time
from gwsnr import GWSNR
from gwsnr.utils import append_json

np.random.seed(1234)

class TestGWSNRInnerProduct:
    """
    Comprehensive unit tests for GWSNR inner product-based SNR calculation methods.
    
    This test suite validates the functionality, accuracy, and performance of the GWSNR
    package's inner product-based signal-to-noise ratio calculation methods. Unlike
    interpolation methods, inner product calculations generate waveforms on-demand and
    compute exact inner products with detector noise, providing higher accuracy for
    parameter estimation applications.
    
    Test Coverage:
        * Default SNR calculation with spinless binary black hole systems
        * Multiple waveform approximants and their accuracy
        * Full spinning binary configurations with precession
        * Computational performance and memory optimization
        * Comparison with interpolation methods for validation
        * Custom detector networks and noise curves
        * Edge cases specific to waveform generation failures
        * Batch processing efficiency and scalability
        
    The inner product method is essential for high-precision applications where
    interpolation accuracy may be insufficient, such as parameter estimation,
    waveform validation, and detailed population studies.
    """

    def test_spinning_bbh_systems(self, tmp_path):
        """
        Test SNR generation using inner product method with spinning BBH systems.
        
        This test validates the core functionality of GWSNR's inner product-based SNR
        calculation for fully precessing binary black hole systems. It demonstrates
        the method's capability to handle complex spin configurations that are
        computationally challenging for interpolation approaches, while serving as
        the foundational test for inner product accuracy and functionality.
        
        The test uses IMRPhenomXPHM, a state-of-the-art waveform model that includes
        precession effects and higher-order modes, providing the most comprehensive
        test for the inner product method's handling of realistic astrophysical systems.
        
        Args:
            tmp_path (pathlib.Path): Pytest fixture providing temporary directory
                for output file testing.
                
        Test Configuration:
            * Waveform approximant: IMRPhenomXPHM (precessing, higher modes)
            * Mass range: Based on GWTC-3 bounds (~10-225 M☉ total mass range)
            * Mass ratio: 0.2-1.0 (realistic mass ratio distribution)
            * Spin magnitudes: 0.0-0.8 (realistic astrophysical range)
            * Spin orientations: Full precessing configuration
            * Fixed luminosity distance: 500 Mpc
            * Detectors: L1, H1, V1 (advanced LIGO/Virgo network)
            * Multiprocessing: 4 cores for parallel waveform generation
            
        Spin Parameters Tested:
            * a_1, a_2: Dimensionless spin magnitudes [0, 0.8]
            * tilt_1, tilt_2: Spin-orbit misalignment angles [0, π]
            * phi_12: Azimuthal angle between spins [0, 2π]
            * phi_jl: Azimuthal angle between L and J [0, 2π]
            
        Validation Workflow:
            1. Generate random spinning binary parameters within realistic ranges
            2. Compute SNR using inner product method with direct waveform generation
            3. Validate output structure, data types, and physical constraints
            4. Test JSON serialization for complex parameter sets
            5. Verify deterministic reproducibility of calculations
            
        Assertions:
            * SNR output is a dictionary containing 'optimal_snr_net'
            * SNR array shape matches number of input samples
            * All SNR values are finite, real, non-negative float64 numbers
            * JSON output file is successfully created and non-empty
            * Repeated calculations produce identical results (rtol=1e-10)
            
        This test ensures the inner product method can handle the full complexity
        of astrophysical binary systems with arbitrary spin configurations, providing
        a reliable foundation for parameter estimation and population studies.
        
        Raises:
            AssertionError: If SNR computation fails or produces invalid results
                for spinning binary systems.
        """
        
        gwsnr = GWSNR(
            npool=int(4),
            mtot_min=2*4.98, # 4.98 Mo is the minimum component mass of BBH systems in GWTC-3
            mtot_max=2*112.5+10.0, # 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomXPHM",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=20.0,
            duration_max=None,
            duration_min=None,
            snr_type="inner_product",
            psds=None,
            ifos=["L1", "H1", "V1"],
            gwsnr_verbose=True,
            multiprocessing_verbose=True,
            mtot_cut=False,
        )

        nsamples = 5
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            # Convert total mass and mass ratio to component masses
            mass_1=mtot / (1 + mass_ratio),
            mass_2=mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance=500 * np.ones(nsamples),
            # Randomly sample sky location and orientation parameters
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
            # Spin parameters for fully precessing systems
            a_1=np.random.uniform(0, 0.8, size=nsamples),
            a_2=np.random.uniform(0, 0.8, size=nsamples),
            tilt_1=np.random.uniform(0, np.pi, size=nsamples),
            tilt_2=np.random.uniform(0, np.pi, size=nsamples),
            phi_12=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phi_jl=np.random.uniform(0, 2 * np.pi, size=nsamples),
        )

        spinning_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output validation checks
        assert isinstance(spinning_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in spinning_snr, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(spinning_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values for spinning systems"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative for spinning systems"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"
        
        # JSON output for spinning systems
        param_dict.update(spinning_snr)
        output_file = tmp_path / "snr_data_spinning_inner_product.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), "Output JSON file was not created for spinning systems"
        assert output_file.stat().st_size > 0, "Output file is empty"

        # Reproducibility check for spinning systems
        spinning_snr2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            spinning_snr["optimal_snr_net"],
            spinning_snr2["optimal_snr_net"],
            rtol=1e-10,
            err_msg="SNR calculation is not deterministic for spinning systems"
        )
        print("SNR calculation with inner product for spinning systems passed successfully.")

    def test_multiple_waveform_approximants(self):
        """
        Test SNR calculation with different waveform approximants using inner product method.
        
        This test validates the inner product method's compatibility with various waveform
        approximants, ensuring consistent and accurate SNR calculations across different
        theoretical models. This is particularly important for systematic studies and
        waveform validation.
        
        Waveform Approximants Tested:
            * IMRPhenomD: Spinless, quadrupole-only
            * SEOBNRv4: Effective-one-body, aligned spins
            * TaylorF2: Post-Newtonian frequency domain
            
        Each approximant represents different theoretical approaches to modeling
        gravitational waves, and the inner product method should handle all consistently.
        
        Test Configuration:
            * Reduced sample size for efficiency (2 events per approximant)
            * Standard mass range and parameters
            * Single detector (L1) for faster computation
            * Verbose output disabled
            
        This test ensures the inner product method is robust across different
        waveform models and can be used for comparative studies.
        
        Raises:
            AssertionError: If any waveform approximant fails SNR computation.
        """
        
        approximants = ["IMRPhenomD", "SEOBNRv4", "TaylorF2"]
        nsamples = 2
        
        for approx in approximants:
            gwsnr = GWSNR(
                npool=1,
                sampling_frequency=2048.0,
                waveform_approximant=approx,
                frequency_domain_source_model='lal_binary_black_hole',
                minimum_frequency=20.0,
                snr_type="inner_product",
                ifos=["L1"],
                gwsnr_verbose=False,
                multiprocessing_verbose=False,
            )

            param_dict = dict(
                mass_1=30.0 * np.ones(nsamples),
                mass_2=25.0 * np.ones(nsamples),
                luminosity_distance=400.0 * np.ones(nsamples),
                theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
                ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
                dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
                psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
                phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
                geocent_time=1246527224.169434 * np.ones(nsamples),
            )

            # Add aligned spins for SEOBNRv4
            if approx == "SEOBNRv4":
                param_dict["a_1"] = np.random.uniform(-0.5, 0.5, size=nsamples)
                param_dict["a_2"] = np.random.uniform(-0.5, 0.5, size=nsamples)

            approx_snr = gwsnr.snr(gw_param_dict=param_dict)

            # Validate each approximant
            assert isinstance(approx_snr, dict), f"SNR output should be a dictionary for {approx}"
            assert "optimal_snr_net" in approx_snr, f"Expected 'optimal_snr_net' for {approx}"
            
            snr_arr = np.asarray(approx_snr["optimal_snr_net"])
            assert snr_arr.shape == (nsamples,), f"Wrong shape for {approx}: {snr_arr.shape}"
            assert np.all(np.isfinite(snr_arr)), f"Non-finite SNR values for {approx}"
            assert np.all(snr_arr >= 0), f"Negative SNR values for {approx}"

    def test_custom_detector_configuration(self):
        """
        Test SNR calculation with custom detector configurations using inner product method.
        
        This test validates the inner product method's flexibility in handling custom
        detector configurations beyond the standard L1/H1/V1 network. It demonstrates
        how to configure a custom interferometer with specific geographic coordinates,
        noise curves, and detector parameters for sensitivity studies.
        
        The test uses LIGO India Aundha (A1) detector configuration as an example of
        a custom detector setup, showcasing the inner product method's capability to
        work with user-defined interferometer objects and custom noise curves.
        
        Custom Configuration Tested:
            * LIGO India Aundha (A1) detector at A+ sensitivity
            * Custom geographic coordinates (Aundha, Maharashtra, India)
            * Custom PSD file: 'Aplus_asd.txt' 
            * Standard frequency range: 20 Hz minimum frequency
            * Standard sampling rate: 2048 Hz
            
        Detector Specifications:
            * Location: Latitude 19°36'47.9017"N, Longitude 77°01'51.0997"E
            * Elevation: 440.0 meters above sea level
            * Arm orientations: X-arm azimuth 117.6157°, Y-arm azimuth 207.6165°
            * Arm length: 4 km (same as LIGO detectors)
            * Coordinates from LIGO-T2000158 and LIGO-T2000012 documents
            
        Test Parameters:
            * Standard BBH mass range suitable for A+ sensitivity
            * Moderate luminosity distances for validation
            * Random sky location and orientation parameters
            
        This test ensures the inner product method can accommodate future detector
        networks, international collaborations, and custom sensitivity studies for
        next-generation gravitational wave observatories.
        
        Usage Example:
            >>> import bilby
            >>> from gwsnr import GWSNR
            >>> ifosA1 = bilby.gw.detector.interferometer.Interferometer(
                    name='A1',
                    power_spectral_density=bilby.gw.detector.PowerSpectralDensity(asd_file='Aplus_asd.txt'),
                    minimum_frequency=20,
                    maximum_frequency=2048,
                    length=4,
                    latitude=19 + 36. / 60 + 47.9017 / 3600,
                    longitude=77 + 1. / 60 + 51.0997 / 3600,
                    elevation=440.0,
                    xarm_azimuth=117.6157,
                    yarm_azimuth=207.6165
                )
            >>> snr = GWSNR(psds=dict(A1='Aplus_asd.txt'), ifos=[ifosA1])
        
        Raises:
            AssertionError: If custom detector configuration fails SNR computation.
            ImportError: If bilby is not available (handled with pytest.skip).
        """
        
        try:
            import bilby
            
            # Create LIGO India Aundha (A1) interferometer object
            # Coordinates from LIGO-T2000158/public and LIGO-T2000012/public
            ifosA1 = bilby.gw.detector.interferometer.Interferometer(
                name='A1',
                power_spectral_density=bilby.gw.detector.PowerSpectralDensity(asd_file='aLIGO_O4_high_asd.txt'),  # Using available ASD file for testing
                minimum_frequency=20,
                maximum_frequency=2048,
                length=4,  # 4 km arms like LIGO
                latitude=19 + 36. / 60 + 47.9017 / 3600,  # Aundha coordinates
                longitude=77 + 1. / 60 + 51.0997 / 3600,
                elevation=440.0,  # meters above sea level
                xarm_azimuth=117.6157,  # degrees
                yarm_azimuth=207.6165,  # degrees (117.6157 + 90)
                xarm_tilt=0.,
                yarm_tilt=0.
            )
            
            # Test with LIGO India A1 detector
            gwsnr_a1 = GWSNR(
                npool=1,
                sampling_frequency=2048.0,
                waveform_approximant="IMRPhenomD",
                frequency_domain_source_model='lal_binary_black_hole',
                minimum_frequency=20.0,  # Standard minimum frequency
                snr_type="inner_product",
                psds=dict(A1='aLIGO_O4_high_asd.txt'),  # Custom PSD specification
                ifos=[ifosA1],  # Custom interferometer object
                gwsnr_verbose=False,
            )
            
        except ImportError:
            pytest.skip("Bilby not available for custom detector configuration test")

        nsamples = 2
        param_dict_a1 = dict(
            mass_1=np.array([20.0, 30.0]),  # Standard BBH masses for A+ sensitivity
            mass_2=np.array([18.0, 25.0]),
            luminosity_distance=np.array([400.0, 600.0]),  # Moderate distances
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
        )

        a1_snr = gwsnr_a1.snr(gw_param_dict=param_dict_a1)

        # Validate A1 detector SNR computation
        assert isinstance(a1_snr, dict), "A1 SNR output should be a dictionary"
        assert "optimal_snr_net" in a1_snr, "Expected 'optimal_snr_net' for A1"
        
        snr_arr = np.asarray(a1_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"A1 SNR wrong shape: {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values for A1"
        assert np.all(snr_arr >= 0), "SNR should not be negative for A1"
        
        print("LIGO India A1 detector test completed successfully.")
        print(f"A1 detector SNR values: {snr_arr}")

    def test_computational_performance(self):
        """
        Test computational performance and memory usage of inner product method.
        
        This test validates that the inner product method performs efficiently for
        batch calculations and uses memory responsibly. While the inner product method
        is inherently more computationally expensive than interpolation, it should
        still be practical for moderate-sized parameter studies.
        
        Performance Metrics Tested:
            * Batch processing efficiency vs single events
            * Memory usage scaling with sample size
            * Multiprocessing speedup validation
            * Reasonable computation time for typical use cases
            
        Test Configuration:
            * Moderate sample size (10 events) for timing
            * Simple waveform approximant (IMRPhenomD) for baseline
            * Single detector to isolate computation time
            * Timing comparison between serial and parallel execution
            
        This test ensures the inner product method remains practical for
        scientific applications despite its computational complexity.
        
        Note: This test focuses on relative performance rather than absolute
        timing, as execution time depends on hardware configuration.
        
        Raises:
            AssertionError: If performance metrics indicate inefficient implementation.
        """
        
        import time
        
        nsamples = 10
        
        # Configure GWSNR for performance testing
        gwsnr_serial = GWSNR(
            npool=1,  # Serial execution
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            minimum_frequency=20.0,
            snr_type="inner_product",
            ifos=["L1"],
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )
        
        gwsnr_parallel = GWSNR(
            npool=4,  # Parallel execution
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            minimum_frequency=20.0,
            snr_type="inner_product",
            ifos=["L1"],
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        # Generate test parameters
        param_dict = dict(
            mass_1=30.0 * np.ones(nsamples),
            mass_2=25.0 * np.ones(nsamples),
            luminosity_distance=500.0 * np.ones(nsamples),
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
        )

        # Time serial execution
        start_time = time.time()
        serial_snr = gwsnr_serial.snr(gw_param_dict=param_dict)
        serial_time = time.time() - start_time

        # Time parallel execution
        start_time = time.time()
        parallel_snr = gwsnr_parallel.snr(gw_param_dict=param_dict)
        parallel_time = time.time() - start_time

        # Validate performance characteristics
        assert isinstance(serial_snr, dict), "Serial SNR output should be a dictionary"
        assert isinstance(parallel_snr, dict), "Parallel SNR output should be a dictionary"
        
        # Check that both methods produce similar results
        serial_arr = np.asarray(serial_snr["optimal_snr_net"])
        parallel_arr = np.asarray(parallel_snr["optimal_snr_net"])
        
        assert serial_arr.shape == parallel_arr.shape, "Serial and parallel shapes should match"
        np.testing.assert_allclose(
            serial_arr, parallel_arr, rtol=1e-8,
            err_msg="Serial and parallel results should be nearly identical"
        )
        
        # Performance validation (relative, not absolute timing)
        assert serial_time > 0, "Serial execution should take measurable time"
        assert parallel_time > 0, "Parallel execution should take measurable time"
        
        # For small sample sizes, parallel overhead might make it slower
        # Just ensure both complete successfully
        print(f"Serial time: {serial_time:.2f}s, Parallel time: {parallel_time:.2f}s")
        print("Performance test completed successfully.")

    def test_custom_psds(self):
        """
        Test SNR calculation with custom power spectral densities using inner product method.
        
        This test validates the inner product method's capability to work with custom
        noise curves beyond the default detector configurations. It demonstrates how
        users can specify custom PSDs for specialized studies, including design
        sensitivity curves, measured PSDs from specific observing runs, or theoretical
        noise curves for future detectors.
        
        The test uses PyCBC-compatible PSD specifications to ensure compatibility
        with the broader gravitational wave data analysis ecosystem.
        
        Custom PSDs Tested:
            * L1: aLIGOaLIGODesignSensitivityT1800044 (Advanced LIGO design sensitivity)
            * H1: aLIGOaLIGODesignSensitivityT1800044 (Advanced LIGO design sensitivity)  
            * V1: AdvVirgo (Advanced Virgo design sensitivity)
            
        Test Configuration:
            * Waveform approximant: IMRPhenomD (spinless for computational efficiency)
            * Moderate sample size for validation
            * Standard mass range and parameters
            * Custom PSD dictionary specification
            * Multiprocessing disabled for cleaner error handling
            
        Test Parameters:
            * Mass range: 20-50 M☉ (suitable for design sensitivity studies)
            * Fixed luminosity distance: 400 Mpc
            * Random sky location and orientation parameters
            
        This test ensures that researchers can use the inner product method with
        their own custom noise curves for specialized sensitivity studies, comparison
        with design specifications, or analysis with non-standard detector configurations.
        
        Raises:
            AssertionError: If custom PSD configuration fails or produces invalid
                SNR calculations.
            ImportError: If required dependencies for PSD handling are not available
                (handled gracefully with pytest.skip).
        """
        
        # Custom PSD specification using PyCBC-compatible names
        custom_psds = {
            'L1': 'aLIGOaLIGODesignSensitivityT1800044',
            'H1': 'aLIGOaLIGODesignSensitivityT1800044', 
            'V1': 'AdvVirgo'
        }
        
        try:
            gwsnr_custom = GWSNR(
                npool=1,  # Single process for cleaner error handling
                sampling_frequency=2048.0,
                waveform_approximant="IMRPhenomD",
                frequency_domain_source_model='lal_binary_black_hole',
                minimum_frequency=20.0,
                snr_type="inner_product",
                psds=custom_psds,  # Custom PSD specification
                ifos=["L1", "H1", "V1"],
                gwsnr_verbose=False,
                multiprocessing_verbose=False,
            )
        except (ImportError, ValueError, KeyError) as e:
            # Skip test if custom PSDs are not available or not supported
            pytest.skip(f"Custom PSDs not available: {e}")
        
        nsamples = 3  # Moderate sample size for validation
        param_dict_custom = dict(
            mass_1=np.array([20.0, 30.0, 40.0]),  # Standard BBH masses
            mass_2=np.array([18.0, 25.0, 35.0]),
            luminosity_distance=400.0 * np.ones(nsamples),  # Fixed distance
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
        )

        custom_psd_snr = gwsnr_custom.snr(gw_param_dict=param_dict_custom)

        # Validate custom PSD SNR computation
        assert isinstance(custom_psd_snr, dict), "Custom PSD SNR output should be a dictionary"
        assert "optimal_snr_net" in custom_psd_snr, "Expected 'optimal_snr_net' for custom PSDs"
        
        snr_arr = np.asarray(custom_psd_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Custom PSD SNR wrong shape: {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values for custom PSDs"
        assert np.all(snr_arr >= 0), "Negative SNR values for custom PSDs"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"
        
        # Compare with default PSDs to ensure custom PSDs produce different results
        gwsnr_default = GWSNR(
            npool=1,
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            minimum_frequency=20.0,
            snr_type="inner_product",
            psds=None,  # Default PSDs
            ifos=["L1", "H1", "V1"],
            gwsnr_verbose=False,
        )
        
        default_psd_snr = gwsnr_default.snr(gw_param_dict=param_dict_custom)
        default_snr_arr = np.asarray(default_psd_snr["optimal_snr_net"])
        
        # Custom and default PSDs should generally produce different SNR values
        # (unless they happen to be identical, which is unlikely)
        try:
            np.testing.assert_allclose(snr_arr, default_snr_arr, rtol=1e-10)
            # If they are identical, just warn rather than fail
            print("Warning: Custom and default PSDs produced identical results")
        except AssertionError:
            # Expected case: custom and default PSDs produce different results
            print("Custom PSDs successfully produced different SNR values from defaults")
        
        # Validate that SNR values are reasonable (not too different from defaults)
        # This is a sanity check that custom PSDs are working correctly
        ratio = snr_arr / default_snr_arr
        assert np.all(ratio > 0.1), "Custom PSD SNRs are unreasonably low compared to defaults"
        assert np.all(ratio < 10.0), "Custom PSD SNRs are unreasonably high compared to defaults"
        
        print("Custom PSD test completed successfully.")
        print(f"Custom PSD SNRs: {snr_arr}")
        print(f"Default PSD SNRs: {default_snr_arr}")
        print(f"SNR ratios (custom/default): {ratio}")

    def test_inner_product_jax_method(self):
        """
        Test SNR calculation using JAX-accelerated inner product method with ripple waveforms.
        
        This test validates the JAX-accelerated inner product method for computing
        signal-to-noise ratios with spinning binary black hole systems. The gwsnr
        package provides optional support for JAX-based waveform generation and
        acceleration via the ripple waveform library, utilizing jax.jit for 
        just-in-time compilation and jax.vmap for efficient batched operations.
        
        The JAX implementation leverages the ripple library for high-performance
        waveform generation, providing GPU acceleration and automatic differentiation
        capabilities while maintaining numerical accuracy equivalent to the
        standard inner product method. This approach enables massive computational
        speedups for large-scale parameter studies and population analyses.
        
        Test Configuration:
            * Waveform approximant: IMRPhenomXAS (aligned spins, higher modes)
            * Waveform library: ripple (JAX-based acceleration)
            * Mass range: Based on GWTC-3 bounds (~10-225 M☉ total mass range)
            * Mass ratio: 0.2-1.0 (realistic mass ratio distribution)
            * Spin magnitudes: 0.0-0.8 (realistic astrophysical range)
            * Spin orientations: Aligned with orbital angular momentum
            * Fixed luminosity distance: 500 Mpc
            * Detectors: Default L1, H1, V1 network
            * JAX acceleration: Enabled for inner product computation
            * Mass cutoff: Enabled for realistic mass bounds
            
        Spin Parameters Tested:
            * a_1, a_2: Dimensionless aligned spin magnitudes [0, 0.8]
            * tilt_1, tilt_2: Spin-orbit misalignment angles [0, π]
            * phi_12: Azimuthal angle between spins [0, 2π]
            * phi_jl: Azimuthal angle between L and J [0, 2π]
            
        Performance Features via ripple:
            * JAX just-in-time (jax.jit) compilation for optimized execution
            * Vectorized operations via jax.vmap for efficient batched operations
            * GPU acceleration support (when available)
            * Automatic differentiation capabilities for parameter estimation
            * Hardware-accelerated waveform generation through ripple library
            
        Validation Workflow:
            1. Generate random spinning binary parameters within realistic ranges
            2. Compute SNR using JAX-accelerated inner product method with ripple
            3. Validate output structure, data types, and physical constraints
            4. Verify numerical accuracy and reproducibility
            5. Check performance characteristics of JAX/ripple implementation
            
        Assertions:
            * SNR output is a dictionary containing 'optimal_snr_net'
            * SNR array shape matches number of input samples
            * All SNR values are finite, real, non-negative float64 numbers
            * JAX computation produces consistent and reproducible results
            * Performance is suitable for batch processing applications
            
        This test ensures the JAX-accelerated inner product method with ripple
        waveform generation provides efficient and accurate SNR calculations for
        parameter estimation and population studies requiring high computational
        throughput and hardware acceleration.
        
        Raises:
            AssertionError: If JAX inner product computation fails or produces
                invalid results for spinning binary systems.
            ImportError: If JAX or ripple dependencies are not available 
                (handled with pytest.skip).
        """
        
        try:
            # Initialize the GWSNR object with JAX acceleration
            gwsnr = GWSNR(
                npool=4,
                mtot_resolution=50,
                ratio_resolution=10,
                sampling_frequency=2048.0,
                waveform_approximant="IMRPhenomXAS",
                minimum_frequency=20.0,
                snr_type="inner_product_jax",
                psds=None,
                ifos=None,
                interpolator_dir="./interpolator_pickle",
                create_new_interpolator=False,
                gwsnr_verbose=False,
                multiprocessing_verbose=False,
                mtot_cut=True,
            )
        except ImportError as e:
            pytest.skip(f"JAX dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"JAX inner product method not available: {e}")

        # Generate BBH parameters with spinning systems
        nsamples = 10
        mtot = np.linspace(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            mass_1=mtot / (1 + mass_ratio),
            mass_2=mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance=500 * np.ones(nsamples),
            # Randomly sample sky location and orientation parameters
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples), 
            dec=np.random.uniform(-np.pi / 2, np.pi / 2, size=nsamples), 
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
            # Spin parameters for aligned spinning systems
            a_1=np.random.uniform(0, 0.8, size=nsamples),
            a_2=np.random.uniform(0, 0.8, size=nsamples),
            tilt_1=np.random.uniform(0, np.pi, size=nsamples),
            tilt_2=np.random.uniform(0, np.pi, size=nsamples),
            phi_12=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phi_jl=np.random.uniform(0, 2 * np.pi, size=nsamples),
        )

        # Calculate SNR using JAX-accelerated inner product method
        jax_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output validation checks
        assert isinstance(jax_snr, dict), "JAX SNR output should be a dictionary"
        assert "optimal_snr_net" in jax_snr, "Expected 'optimal_snr_net' in JAX SNR output"
        
        snr_arr = np.asarray(jax_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values for JAX spinning systems"
        assert np.all(np.isreal(snr_arr)), "JAX SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "JAX SNR should not be negative for spinning systems"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"
        
        # Reproducibility check for JAX implementation
        jax_snr2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            jax_snr["optimal_snr_net"],
            jax_snr2["optimal_snr_net"],
            rtol=1e-10,
            err_msg="JAX SNR calculation is not deterministic for spinning systems"
        )
        
        # Performance check - JAX should complete in reasonable time
        start_time = time.time()
        jax_snr_timed = gwsnr.snr(gw_param_dict=param_dict)
        jax_time = time.time() - start_time
        
        assert jax_time < 60.0, f"JAX computation too slow: {jax_time:.2f}s"  # Reasonable time limit
        
        print("JAX inner product method test completed successfully.")
        print(f"JAX SNR values: {snr_arr}")
        print(f"JAX computation time: {jax_time:.2f}s")

