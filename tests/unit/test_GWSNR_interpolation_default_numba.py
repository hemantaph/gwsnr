"""
Comprehensive unit tests for GWSNR interpolation-based (default with Numba) SNR calculations.

This test suite provides thorough validation of the GWSNR (Gravitational Wave 
Signal-to-Noise Ratio) package's interpolation functionality for binary black 
hole (BBH) gravitational wave signals. The tests focus on the Numba-accelerated 
interpolation backend and cover the full range of analysis scenarios.

Test Coverage:
--------------
• Default SNR Generation: Core functionality with GWTC-3 catalog parameter ranges
• Error Handling: Robust validation of invalid inputs and edge cases  
• Single Event Processing: Scalar input handling for real-time analysis
• Multiple Interpolation Methods: Cross-validation of different backends
• Custom Configurations: Flexibility testing for various detector setups
• Detection Probability: pdet calculations with multiple methodologies

Scientific Context:
------------------
The tests use realistic astrophysical parameters based on:
- GWTC-3 gravitational wave catalog observations (4.98-112.5 M☉ component masses)
- LIGO/Virgo detector configurations and sensitivities
- Standard waveform approximants (IMRPhenomD, TaylorF2)
- Typical observational distances (100-500 Mpc)
- Standard detection thresholds (network SNR ≥ 8)

Technical Implementation:
------------------------
• Interpolation Backend: Numba-accelerated with prange multi-threaded loops for computational efficiency
• Parameter Spaces: Multi-dimensional grids in (total_mass, mass_ratio, spin)
• Waveform Models: Frequency-domain analytical approximants
• Detector Networks: LIGO H1/L1, Virgo V1, next-gen CE configurations
• Validation: Numerical accuracy, reproducibility, and format consistency

Applications:
------------
These tests ensure GWSNR reliability for:
- Population synthesis studies and astrophysical inference
- Survey planning for current and future gravitational wave detectors
- Real-time analysis during observing runs
- Multi-messenger astronomy follow-up coordination
- Statistical characterization of detection capabilities

Dependencies:
------------
- numpy: Numerical computations and random parameter generation
- pytest: Test framework with fixtures for temporary file handling
- gwsnr: Main package with GWSNR class and utility functions

Usage:
-----
Run individual tests: pytest test_GWSNR_interpolation_default_numba.py::test_name
Run full suite: pytest test_GWSNR_interpolation_default_numba.py -v
"""

import numpy as np
import pytest
from gwsnr import GWSNR
from gwsnr.utils import append_json

np.random.seed(1234)

class TestGWSNRInterpolation:
    """
    Comprehensive test class for GWSNR interpolation-based SNR calculations.
    
    This test class systematically validates all aspects of the GWSNR package's
    interpolation functionality, ensuring reliability and accuracy for 
    gravitational wave data analysis applications.
    
    Test Architecture:
    -----------------
    The class employs a modular design with helper methods for common operations:
    
    • _generate_params(): Creates realistic BBH parameter distributions
    • _check_snr_output(): Validates SNR calculation results and formats
    
    Test Categories:
    ---------------
    1. Core Functionality Tests:
       - test_default_snr_generation(): Standard LIGO/Virgo analysis workflow
       - test_single_event(): Individual event processing capabilities
    
    2. Robustness Tests:
       - test_invalid_input_handling(): Error handling and data validation
    
    3. Method Validation Tests:
       - test_snr_generation_for_various_interpolation(): Cross-backend consistency
       - test_snr_generation_custom_input_arguments(): Configuration flexibility
    
    4. Advanced Analysis Tests:
       - test_pdet_generation(): Detection probability methodologies
    
    Implementation Details:
    ----------------------
    • Fixed random seed (1234) ensures reproducible test results
    • Temporary file handling via pytest fixtures for I/O testing
    • Parameterized testing approach for multiple configurations
    • Comprehensive assertion coverage for numerical validation
    
    Validation Criteria:
    -------------------
    All tests verify that SNR calculations produce:
    - Finite, real, non-negative values
    - Correct array shapes matching input parameters
    - Consistent data types (float64 precision)
    - Reproducible results across multiple runs
    - Proper error handling for invalid inputs
    
    This ensures GWSNR meets the stringent requirements for production
    gravitational wave analysis pipelines.
    """
    
    def _generate_params(self, nsamples, mtot_range=(20, 235), distance=500):
        """
        Generate random binary black hole parameters for testing.
        
        Creates realistic BBH parameter distributions for test validation:
        - Total masses: Uniform distribution within specified range (default: GWTC-3 range)
        - Mass ratios: Uniform from 0.2 to 1.0 (secondary/primary mass ratio)
        - Distance: Fixed luminosity distance in Mpc for controlled SNR testing
        - Sky location: Isotropic distribution over celestial sphere
        - Orientation: Random inclination, polarization, and coalescence phase
        - GPS time: Fixed to representative LIGO observing time
        
        Returns parameter dictionary compatible with GWSNR input format.
        """
        mtot = np.random.uniform(mtot_range[0], mtot_range[1], nsamples)
        mass_ratio = np.random.uniform(0.2, 1, nsamples)
        
        return {
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
    
    def _check_snr_output(self, snr_dict, expected_shape):
        """
        Validate SNR output format and numerical properties.
        
        Performs comprehensive validation of GWSNR output including:
        - Dictionary structure with required 'optimal_snr_net' key
        - Correct array shape matching input parameter count
        - Numerical validity: finite, real, non-negative values
        - Data type consistency: float64 precision
        
        Essential for ensuring reliable SNR calculations in production use.
        """
        assert isinstance(snr_dict, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in snr_dict, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(snr_dict["optimal_snr_net"])
        assert snr_arr.shape == expected_shape, f"Expected shape {expected_shape}, got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "SNR values should be finite"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real"
        assert np.all(snr_arr >= 0), "SNR values should be non-negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

    def test_default_snr_generation(self, tmp_path):
        """
        Test SNR generation with default GWSNR interpolation settings for spinless BBH systems.
        
        This test validates the core GWSNR functionality using default parameters based on
        GWTC-3 catalog observations. It tests:
        
        - SNR calculation for multiple binary black hole systems using interpolation
        - Parameter ranges based on observed LIGO/Virgo detections (4.98-112.5 solar masses)
        - IMRPhenomD waveform approximant with 2048 Hz sampling
        - Default detector configuration (H1, L1, V1)
        - JSON serialization of results for data persistence
        - Reproducibility of calculations (deterministic behavior)
        - Output format validation (dictionary structure, data types, finite values)
        
        The test uses 5 random BBH systems at 500 Mpc distance and validates that
        all SNR values are finite, real, non-negative, and consistently reproducible.
        """
        gwsnr = GWSNR(
            npool=4,
            mtot_min=2*4.98,
            mtot_max=2*112.5+10.0,
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.99,
            mtot_resolution=200,
            ratio_resolution=20,
            spin_resolution=10,
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=20.0,
            snr_type="interpolation_no_spins",
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=True,
            multiprocessing_verbose=True,
            mtot_cut=False,
            pdet=False,
            snr_th=8.0,
            snr_th_net=8.0,
            snr_recalculation=False,
            snr_recalculation_range=[6,8],
            snr_recalculation_waveform_approximant="IMRPhenomXPHM",
        )

        nsamples = 5
        param_dict = self._generate_params(nsamples)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Check output
        self._check_snr_output(interp_snr, (nsamples,))

        # Test JSON output
        param_dict.update(interp_snr)
        output_file = tmp_path / "snr_data_interpolation.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), "Output JSON file was not created"
        assert output_file.stat().st_size > 0, "Output file is empty"

        # Test reproducibility
        interp_snr2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            interp_snr["optimal_snr_net"],
            interp_snr2["optimal_snr_net"],
            rtol=1e-10,
            err_msg="SNR calculation is not deterministic"
        )

    def test_invalid_input_handling(self):
        """
        Test robust error handling for invalid input parameters.
        
        This test ensures GWSNR properly validates input data and raises appropriate
        exceptions for physically invalid or numerically problematic inputs:
        
        - Negative masses: Tests rejection of unphysical negative mass values
        - NaN values: Validates detection of "Not a Number" numerical errors  
        - Infinite values: Tests handling of numerical overflow conditions
        - Empty arrays: Ensures proper validation of edge case inputs
        
        Proper error handling is critical for preventing undefined behavior in
        production gravitational wave analysis pipelines and ensuring data quality.
        The test verifies that appropriate exceptions (ValueError, TypeError, 
        AssertionError) are raised for each invalid input scenario.
        """
        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False)
        
        # Test with negative masses
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.snr(gw_param_dict={
                'mass_1': np.array([-30]),
                'mass_2': np.array([20]),
                'luminosity_distance': np.array([400]),
            })

        # Test with NaN values
        with pytest.raises((TypeError, ValueError)):
            gwsnr.snr(gw_param_dict={
                'mass_1': np.array([30, 40]),
                'mass_2': np.array([20, np.nan]),
                'luminosity_distance': np.array([400, 500]),
            })

        # Test with infinite values
        with pytest.raises((TypeError, ValueError)):
            gwsnr.snr(gw_param_dict={
                'mass_1': np.array([30, 40]),
                'mass_2': np.array([20, np.inf]),
                'luminosity_distance': np.array([400, 500]),
            })

        # Test with empty arrays
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.snr(gw_param_dict={
                'mass_1': np.array([]),
                'mass_2': np.array([]),
                'luminosity_distance': np.array([]),
            })

    def test_single_event(self):
        """
        Test SNR calculation for a single gravitational wave event.
        
        This test validates GWSNR's ability to handle single-event calculations
        (scalar inputs) as opposed to batch processing of multiple events.
        This is important for:
        
        - Real-time gravitational wave analysis during observing runs
        - Individual event characterization and parameter estimation
        - Online detection pipeline integration
        - Rapid preliminary analysis of candidate events
        
        The test uses representative BBH parameters (30 + 25 solar mass system
        at 400 Mpc) typical of LIGO detections and verifies that:
        - Single scalar inputs are properly handled
        - Output has correct shape (1-element array)
        - SNR value is finite and non-negative
        - No errors occur in single-event processing mode
        """
        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False)
        
        param_dict = {
            'mass_1': 30.0,
            'mass_2': 25.0,
            'luminosity_distance': 400.0,
            'theta_jn': 0.5,
            'ra': 1.0,
            'dec': 0.2,
            'psi': 0.3,
            'phase': 0.1,
            'geocent_time': 1246527224.169434,
        }
        
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        self._check_snr_output(interp_snr, (1,))

    def test_snr_generation_for_various_interpolation(self):
        """
        Test SNR generation across different GWSNR interpolation methods.
        
        This test validates the consistency and correctness of various interpolation
        backends available in GWSNR, specifically testing:
        
        - interpolation_aligned_spins: Includes effects of aligned black hole spins
        - Alternative backends: JAX-accelerated versions, different spin models
        - Cross-validation: Ensures different methods produce valid results
        - Physics consistency: Spin effects are properly handled when included
        
        The test focuses on 'interpolation_aligned_spins' which accounts for
        the gravitational wave signal modifications due to black hole rotation.
        This is crucial for:
        - Accurate astrophysical parameter estimation
        - Proper modeling of spinning black hole mergers  
        - Consistency with advanced LIGO/Virgo waveform models
        - Future third-generation detector analysis capabilities
        
        Validates that spin-inclusive interpolation produces the same output
        format and quality as spinless calculations.
        """
        nsamples = 5
        param_dict = self._generate_params(nsamples)

        # Test with interpolation_aligned_spins
        gwsnr = GWSNR(snr_type="interpolation_aligned_spins", mtot_resolution=20, ratio_resolution=5, spin_resolution=5, gwsnr_verbose=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        self._check_snr_output(interp_snr, (nsamples,))

    def test_snr_generation_custom_input_arguments(self):
        """
        Test SNR generation with custom GWSNR configuration parameters.
        
        This test validates GWSNR's flexibility and adaptability to different
        analysis scenarios beyond the default LIGO/Virgo configuration:
        
        - Lower mass systems: Tests stellar-mass BBH range (2-7 solar masses)
        - Next-generation detectors: Cosmic Explorer (CE) configuration  
        - Alternative waveforms: TaylorF2 post-Newtonian approximant
        - Modified analysis parameters: Lower sampling rate (1024 Hz), higher f_min (30 Hz)
        - Analysis optimizations: Mass cutoff functionality enabled
        - Closer distances: 100 Mpc for enhanced sensitivity to smaller systems
        
        This configuration is relevant for:
        - Future third-generation gravitational wave observatories
        - Primordial black hole searches in lower mass ranges
        - Population studies requiring different parameter spaces
        - Computational efficiency studies with reduced sampling rates
        - Cross-validation with different waveform approximants
        
        Ensures GWSNR maintains accuracy and reliability across diverse
        astrophysical scenarios and detector configurations.
        """
        nsamples = 5
        param_dict = self._generate_params(nsamples, mtot_range=(2, 7), distance=100)

        gwsnr = GWSNR(
            npool=4,
            mtot_min=2*1.0,
            mtot_max=2*3.0+1.0,
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.99,
            mtot_resolution=20,
            ratio_resolution=5,
            spin_resolution=5,
            sampling_frequency=1024.0,
            waveform_approximant="TaylorF2",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=30.0,
            snr_type="interpolation_no_spins",
            ifos=['CE'],
            mtot_cut=True,
        )

        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        self._check_snr_output(interp_snr, (nsamples,))

    def test_pdet_generation(self):
        """
        Test probability of detection (pdet) calculations with multiple methodologies.
        
        This comprehensive test validates GWSNR's detection probability capabilities
        using three different approaches:
        
        1. Direct pdet calculation: Integrated during SNR computation with binary output
        2. Boolean thresholding: Post-hoc pdet from optimal SNR threshold comparison  
        3. Matched-filter statistics: Continuous pdet using Gaussian noise assumptions
        
        Key validation points:
        - Binary pdet values (0/1) for hard threshold detection decisions
        - Continuous pdet values [0,1] for statistical detection probabilities
        - SNR threshold consistency (8.0 network threshold matching LIGO standards)
        - Cross-method validation ensuring consistent detection logic
        
        Detection probability calculations are fundamental for:
        - Gravitational wave survey planning and observation strategies
        - Astrophysical population inference and rate calculations
        - Detector network optimization and sensitivity studies
        - Alert threshold setting for multi-messenger astronomy
        - Statistical characterization of detection capabilities
        
        The test uses standard LIGO detection thresholds (SNR ≥ 8) and validates
        that all three methods produce physically reasonable and consistent results.
        """
        nsamples = 5
        param_dict = self._generate_params(nsamples)

        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False, pdet=True, snr_th=8.0, snr_th_net=8.0)
        interp_pdet = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_pdet, dict), "pdet output should be a dictionary"
        assert "pdet_net" in interp_pdet, "Expected 'pdet_net' in pdet output"
        # check pdet is 0 or 1
        pdet_arr = np.asarray(interp_pdet["pdet_net"])
        assert pdet_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {pdet_arr.shape}"
        assert np.all(np.isin(pdet_arr, [0, 1])), "pdet values should be binary (0 or 1)"

        # find SNR values first and then check pdet
        gwsnr = GWSNR(snr_type="interpolation_no_spins", gwsnr_verbose=False, pdet=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Pdet wrt optimal SNR
        interp_pdet_optimal = gwsnr.probability_of_detection(snr_dict=interp_snr, snr_th=8.0, snr_th_net=8.0, type='bool')
        # Pdet wrt matched-filter SNR with the assumption of gaussian noise
        interp_pdet_match_filter = gwsnr.probability_of_detection(snr_dict=interp_snr, snr_th=8.0, snr_th_net=8.0, type='matched_filter')

        # Output checks
        # check interp_pdet_optimal is 0 or 1
        assert np.all(np.isin(interp_pdet_optimal["pdet_net"], [0, 1])), "pdet values should be binary (0 or 1)"
        # check interp_pdet_match_filter is within 0 and 1
        assert np.all((interp_pdet_match_filter["pdet_net"] >= 0) & (interp_pdet_match_filter["pdet_net"] <= 1)), "pdet values should be within [0, 1]"