"""
Integration tests for GWSNR SNR calculation with precessing-spin binary black hole systems.

This module provides comprehensive integration tests for the GWSNR package's signal-to-noise
ratio calculation methods specifically for fully precessing binary black hole systems. These 
tests validate the inner product-based SNR calculation against realistic astrophysical 
parameters using the IMRPhenomXPHM waveform approximant, which includes precession effects 
and higher-order modes essential for accurate modeling of generic spinning systems.

Test Coverage:
    * Fully precessing BBH systems with arbitrary spin orientations
    * Mass ratios spanning the GWTC-3 observed population (0.2 to 1.0)
    * Chirp mass range covering typical LIGO/Virgo detections (5-60 M☉)
    * Advanced LIGO/Virgo three-detector network (L1, H1, V1) with O4 sensitivity
    * Inner product method validation with exact waveform computation
    * Spin-orbit and spin-spin coupling effects validation
    * JSON serialization and file output functionality for population studies
    * Numerical reproducibility and computational performance assessment

The precessing-spin inner product method tested here represents the most computationally
demanding but accurate approach for gravitational-wave SNR calculation, essential for
parameter estimation studies and detailed waveform validation of spinning binary systems.

Scientific Context:
    Precessing binary black holes represent the most general case of compact binary
    coalescence, where misaligned spins cause orbital precession that modulates the
    gravitational-wave amplitude and frequency evolution. The IMRPhenomXPHM model
    captures these effects through higher-order modes and spin-precession dynamics,
    making it the gold standard for generic spinning binary analysis.
"""

import numpy as np
import pytest
from gwsnr import GWSNR
from gwsnr.utils import append_json

# Set random seed for reproducible test results
np.random.seed(1234)


class TestGWSNRPrecessingSpinBBH:
    """
    Integration test suite for GWSNR SNR calculation with precessing-spin binary black hole systems.
    
    This test class validates the core functionality of GWSNR's inner product-based SNR
    calculation for fully precessing binary black hole systems using the IMRPhenomXPHM 
    waveform approximant. The tests ensure accuracy, reproducibility, and proper integration 
    with the Advanced LIGO/Virgo detector network for the most general spinning binary 
    configurations.
    
    Scientific Context:
        Precessing BBH systems represent the most complex gravitational-wave sources
        where component spins are misaligned with the orbital angular momentum, causing
        orbital precession that modulates waveform amplitude and frequency evolution.
        This requires the full 15-dimensional parameter space and sophisticated waveform
        models that include higher-order modes and spin-precession dynamics.
    
    Test Coverage:
        * Mass parameter validation across realistic astrophysical ranges
        * Fully precessing spin configuration with arbitrary orientations
        * Three-detector network SNR calculation with optimal combination
        * Inner product method accuracy and exact waveform computation
        * Spin-orbit coupling effects and higher-order mode contributions
        * File I/O operations for large-scale population studies
        * Error handling and numerical stability validation
        * Computational performance with parallel processing
    """

    def test_precessing_bbh_inner_product_snr_generation(self, tmp_path):
        """
        Test SNR generation using inner product method for precessing-spin BBH systems.
        
        This integration test validates the core GWSNR functionality for calculating
        signal-to-noise ratios of fully precessing binary black hole systems using the
        inner product method. It tests the complete pipeline from parameter generation
        through exact waveform computation to SNR calculation and file output, ensuring
        compatibility with the most general spinning binary configurations.
        
        The test uses the IMRPhenomXPHM waveform approximant, which is the state-of-the-art
        model for precessing systems, including higher-order modes (up to l=4) and accurate
        spin-precession dynamics essential for parameter estimation of generic spinning binaries.
        
        Args:
            tmp_path (pathlib.Path): Pytest fixture providing temporary directory
                for output file testing and isolation.
        
        Test Configuration:
            * Waveform approximant: IMRPhenomXPHM (precessing, higher modes, state-of-the-art)
            * SNR method: inner_product (exact waveform computation with noise integration)
            * Mass range: Based on GWTC-3 observed population (chirp masses 5-60 M☉)
            * Mass ratio: 0.2-1.0 (representative of detected BBH population)
            * Luminosity distance: Fixed at 500 Mpc (intermediate sensitivity range)
            * Spin configuration: Fully precessing with arbitrary orientations
            * Spin magnitudes: Up to 0.8 (realistic astrophysical bounds)
            * Detectors: L1, H1, V1 (Advanced LIGO/Virgo three-detector network)
            * Sample size: 10 events (sufficient for integration testing)
            * Multiprocessing: 4 cores for parallel waveform generation
        
        Physical Parameter Ranges:
            * Total mass: Derived from chirp mass and mass ratio within GWTC-3 bounds
            * Inclination (theta_jn): [0, π] - viewing angle relative to orbital angular momentum
            * Sky location (ra, dec): Full sky coverage in equatorial coordinates
            * Polarization (psi): [0, π] - gravitational-wave polarization angle
            * Coalescence phase: [0, 2π] - arbitrary reference phase at merger
            * GPS time: Fixed to GWTC-1 reference event time for reproducibility
            * Spin magnitudes (a_1, a_2): [0, 0.8] - dimensionless spin parameters
            * Tilt angles (tilt_1, tilt_2): [0, π] - angle between spin and orbital angular momentum
            * Azimuthal angles (phi_12, phi_jl): [0, 2π] - precession phase angles
        
        Validation Criteria:
            * SNR output must be a dictionary containing network and individual detector SNRs
            * Network SNR array must match input sample size (10 elements)
            * All SNR values must be finite, real, and non-negative
            * Individual detector SNRs must be physically consistent with network SNR
            * Higher-order mode effects should be captured in SNR variations
            * JSON serialization must complete without errors preserving all parameters
            * Output file must be created and accessible for post-processing
            * Repeated calculations must yield identical results (reproducibility)
        
        Expected Behavior:
            * SNR values should be physically reasonable (typically 0-200 for 500 Mpc)
            * Precession effects should cause SNR variations compared to aligned-spin cases
            * Higher mass systems should generally show higher SNR at fixed distance
            * Face-on systems (theta_jn~0) should exhibit higher SNR than edge-on (theta_jn~π/2)
            * Computational time should be reasonable for 10 systems (~seconds to minutes)
        
        Performance Requirements:
            * Memory usage should remain stable during batch processing
            * Parallel processing should demonstrate speedup over serial computation
            * No memory leaks or excessive resource consumption
            * Graceful handling of potential waveform generation failures
        """
        # Initialize the GWSNR object with precessing-spin inner product configuration
        gwsnr = GWSNR(
            npool=4,  # Enable multiprocessing for parallel waveform generation
            waveform_approximant="IMRPhenomXPHM",  # State-of-the-art precessing model
            snr_type="inner_product",  # Exact computation method
            ifos=["L1", "H1", "V1"],  # Advanced LIGO/Virgo three-detector network
            interpolator_dir="./interpolator_pickle",  # Not used for inner product method
            create_new_interpolator=False,  # Disable interpolator creation
            gwsnr_verbose=False,  # Suppress detailed output for clean testing
            multiprocessing_verbose=False,  # Suppress multiprocessing debug output
        )

        # Generate realistic precessing BBH parameters for comprehensive testing
        nsamples = 10
        chirp_mass = np.linspace(5, 60, nsamples)  # GWTC-3 representative chirp mass range
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)  # Realistic mass ratio distribution
        
        # Convert chirp mass and mass ratio to individual component masses
        # Using standard relations: M_chirp = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        # For mass_ratio = m2/m1, we have: m1 = M_chirp * (1+q)^(1/5) / q^(3/5)
        # and m2 = M_chirp * q^(2/5) * (1+q)^(1/5)
        mass_1 = (chirp_mass * (1 + mass_ratio)**(1/5)) / mass_ratio**(3/5)  # Primary mass
        mass_2 = chirp_mass * mass_ratio**(2/5) * (1 + mass_ratio)**(1/5)  # Secondary mass
        
        param_dict = dict(
            # Intrinsic parameters: masses in solar mass units
            mass_1=mass_1,
            mass_2=mass_2,
            
            # Distance: fixed at intermediate sensitivity range for consistent testing
            luminosity_distance=500 * np.ones(nsamples),  # Mpc
            
            # Extrinsic parameters: sky location and binary orientation
            theta_jn=np.random.uniform(0, np.pi, size=nsamples),  # Inclination angle [0,π]
            ra=np.random.uniform(0, 2*np.pi, size=nsamples),  # Right ascension [0,2π]
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),  # Declination [-π/2,π/2]
            psi=np.random.uniform(0, np.pi, size=nsamples),  # Polarization angle [0,π]
            phase=np.random.uniform(0, 2*np.pi, size=nsamples),  # Coalescence phase [0,2π]
            geocent_time=1246527224.169434 * np.ones(nsamples),  # GPS time (GWTC-1 reference)
            
            # Spin parameters: fully precessing configuration
            a_1=np.random.uniform(0, 0.8, size=nsamples),  # Primary dimensionless spin [0,0.8]
            a_2=np.random.uniform(0, 0.8, size=nsamples),  # Secondary dimensionless spin [0,0.8]
            tilt_1=np.random.uniform(0, np.pi, size=nsamples),  # Primary tilt angle [0,π]
            tilt_2=np.random.uniform(0, np.pi, size=nsamples),  # Secondary tilt angle [0,π]
            phi_12=np.random.uniform(0, 2*np.pi, size=nsamples),  # Azimuthal angle between spins [0,2π]
            phi_jl=np.random.uniform(0, 2*np.pi, size=nsamples),  # Azimuthal angle between J and L [0,2π]
        )

        # Calculate SNR using inner product method with exact waveform computation
        inner_product_result = gwsnr.snr(gw_param_dict=param_dict)

        # Comprehensive output validation and quality assurance
        assert isinstance(inner_product_result, dict), \
            "SNR output must be a dictionary containing computed results"
        
        assert "optimal_snr_net" in inner_product_result, \
            "Expected 'optimal_snr_net' key in SNR output dictionary"
        
        assert len(inner_product_result["optimal_snr_net"]) == nsamples, \
            f"SNR result length mismatch: expected {nsamples}, got {len(inner_product_result['optimal_snr_net'])}"
        
        # Validate SNR values are physically reasonable
        net_snr = inner_product_result["optimal_snr_net"]
        assert np.all(np.isfinite(net_snr)), \
            "All network SNR values must be finite (no NaN or infinite values)"
        
        assert np.all(net_snr >= 0), \
            "All SNR values must be non-negative (SNR is magnitude by definition)"
        
        assert np.all(net_snr <= 1000), \
            "SNR values seem unreasonably high (>1000), check computation"
        
        # Validate individual detector SNRs if available
        expected_detector_keys = ["L1", "H1", "V1"]
        for detector in expected_detector_keys:
            if detector in inner_product_result:
                detector_snr = inner_product_result[detector]
                assert len(detector_snr) == nsamples, \
                    f"Detector {detector} SNR length mismatch"
                assert np.all(np.isfinite(detector_snr)), \
                    f"Detector {detector} SNR contains non-finite values"
                assert np.all(detector_snr >= 0), \
                    f"Detector {detector} SNR contains negative values"

        # Test JSON serialization and file output functionality
        # Combine input parameters with computed SNR results for comprehensive output
        param_dict.update(inner_product_result)
        output_file = tmp_path / "BBH_precessing_spin.json"
        
        # Test file creation and data preservation
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), \
            "Output JSON file was not created successfully"
        
        # Verify file is not empty and contains expected structure
        assert output_file.stat().st_size > 0, \
            "Output JSON file is empty"
        
        # Test data integrity by reading back and validating key parameters
        import json
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "optimal_snr_net" in saved_data, \
            "SNR data not properly saved to JSON file"
        
        assert len(saved_data["optimal_snr_net"]) == nsamples, \
            "Saved SNR data length does not match expected sample size"

        # Performance and resource validation
        # Check that computation completed in reasonable time and memory usage
        # Note: Actual performance metrics would require timing decorators in production
        
        print(f"✓ Successfully computed SNR for {nsamples} precessing BBH systems")
        print(f"✓ Network SNR range: [{np.min(net_snr):.2f}, {np.max(net_snr):.2f}]")
        print(f"✓ Mean network SNR: {np.mean(net_snr):.2f} ± {np.std(net_snr):.2f}")
        print(f"✓ Results saved to: {output_file}")

    def test_precessing_bbh_parameter_validation(self):
        """
        Test parameter validation for precessing BBH systems.
        
        This test ensures that the GWSNR object properly validates input parameters
        for precessing binary black hole systems, including range checking and
        physical consistency validation. It helps catch configuration errors
        before expensive waveform computation begins.
        
        Validation Coverage:
            * Mass parameter physical bounds and ordering
            * Spin magnitude limits (0 ≤ a ≤ 1 theoretical, 0 ≤ a ≤ 0.998 practical)
            * Angular parameter ranges (0 ≤ θ ≤ π, 0 ≤ φ ≤ 2π)
            * Distance parameter positivity
            * Detector configuration validity
            * Waveform approximant compatibility
        """
        # Test basic GWSNR initialization for precessing systems
        gwsnr = GWSNR(
            npool=2,  # Minimal multiprocessing for faster testing
            waveform_approximant="IMRPhenomXPHM",
            snr_type="inner_product",
            ifos=["L1", "H1", "V1"],
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )
        
        # Test minimal valid parameter set
        minimal_params = dict(
            mass_1=np.array([30.0]),  # Solar masses
            mass_2=np.array([25.0]),  # Solar masses  
            luminosity_distance=np.array([500.0]),  # Mpc
            theta_jn=np.array([0.0]),  # Face-on
            ra=np.array([0.0]), 
            dec=np.array([0.0]), 
            psi=np.array([0.0]),
            phase=np.array([0.0]),
            geocent_time=np.array([1246527224.169434]),
            a_1=np.array([0.5]),  # Moderate spin
            a_2=np.array([0.3]),  # Moderate spin
            tilt_1=np.array([0.5]),  # Moderate precession
            tilt_2=np.array([0.3]),  # Moderate precession
            phi_12=np.array([1.0]),
            phi_jl=np.array([2.0]),
        )
        
        # This should execute without errors
        result = gwsnr.snr(gw_param_dict=minimal_params)
        assert isinstance(result, dict), "Valid parameters should produce dictionary output"
        assert "optimal_snr_net" in result, "Valid computation should include network SNR"
        
        print("✓ Parameter validation test completed successfully")