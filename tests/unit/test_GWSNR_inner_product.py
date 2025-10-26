"""
Unit tests for GWSNR inner-product based SNR calculation methods.

Test Coverage:
--------------
- Inner product SNR for spin-precessing BBH systems, using IMRPhenomXPHM
- Multiple waveform approximants (IMRPhenomD, TaylorF2)
- Custom detector configurations and PSDs
- Serial vs parallel multiprocessing performance
- Output validation and reproducibility

Usage:
-----
pytest tests/unit/test_GWSNR_inner_product.py -v -s
pytest tests/unit/test_GWSNR_inner_product.py::TestGWSNRInnerProduct::test_name -v -s
"""

import os
import numpy as np
import time
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

np.random.seed(1234)

# Default GWSNR configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
CONFIG = {
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
    'waveform_approximant': "IMRPhenomXPHM",    # Waveform model for BBH systems
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "inner_product",  # Use interpolation with aligned spins
    'interpolator_dir': "../interpolator_pickle", # Directory for saved interpolators
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

    # SNR recalculation settings
    'snr_recalculation': False,
    'snr_recalculation_range': [4, 12],
    'snr_recalculation_waveform_approximant': "IMRPhenomXPHM",
}

class TestGWSNRInnerProduct(CommonTestUtils):
    """
    Test suite for GWSNR inner product-based SNR calculations.
    """

    def test_spinning_precessing_bbh_systems(self):
        """
        Tests
        -----
        - Spin-precessing BBH systems using IMRPhenomXPHM approximant
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - JSON output file creation and content verification
        - Reproducibility: Deterministic results across multiple runs
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = CONFIG.copy()
        config['gwsnr_verbose'] = True
        
        # Initialize GWSNR instance with test configuration
        gwsnr = GWSNR(**config)

        # Generate test parameters for BBH events with aligned spins
        nsamples = 8  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=True    # Include precessing spins
        )
        
        # Calculate SNR values and save results to JSON file
        output_file = "snr_data_interpolation.json"
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        # Validate that output has correct structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

        # Verify that JSON output file was created successfully
        assert os.path.exists(output_file), "Output JSON file was not created"
        assert os.path.getsize(output_file) > 0, "Output file is empty"

        # Test reproducibility
        snr_result2 = gwsnr.optimal_snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            snr_result["snr_net"], # Network SNR from first calculation
            snr_result2["snr_net"], # Network SNR from second calculation
            rtol=1e-10, # Very tight tolerance for reproducibility
            err_msg="Non-deterministic SNR calculation"
        )

    def test_multiple_waveform_approximants(self):
        """
        Tests
        -----
        - Compatibility with different waveform approximants
            - IMRPhenomD, TaylorF2
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """
        # Configure GWSNR with reduced verbosity for cleaner test output
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False  # Suppress log messages during error testing
        config['ifos'] = ["L1"]          # Use single detector for simplicity

        approximants = ["IMRPhenomD", "TaylorF2"]
        nsamples = 8
        
        for approx in approximants:
            config['waveform_approximant'] = approx
            gwsnr = GWSNR(**config)
            
            # Generate parameters with spins for SEOBNRv4

            if approx == "SEOBNRv4":
                param_dict = self._generate_params(
                    nsamples, 
                    event_type='bbh',        # Binary black hole events
                    spin_zero=False,         # Include aligned spin parameters
                    spin_precession=True    # Include precessing spins
                )
            else:
                param_dict = self._generate_params(
                    nsamples, 
                    event_type='bbh',        # Binary black hole events
                    spin_zero=False,         # Include aligned spin parameters
                    spin_precession=False    # No precessing spins
                )

            snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
            # Validate that output has correct structure and numerical properties
            self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

    def test_custom_detector_configuration(self):
        """
        Tests
        -----
        - SNR calculation with custom detector configuration (LIGO India A1)
        - SNR calculation with custom PSDs (from pycbc) for standard detectors
        - Output validation: dictionary structure, data types, shapes, numerical properties
        """

        # Generate test parameters for BBH events with aligned spins
        nsamples = 8  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=True     # Include precessing spins
        )

        ######################################################
        # Test custom detector configuration (LIGO India A1) #
        ######################################################
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
        
        # Configure GWSNR with reduced verbosity for cleaner test output
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False  # Suppress log messages during error testing
        config['ifos'] = [ifo_a1]        # Use single custom detector
        config['psds'] = {'A1': 'aLIGO_O4_high_asd.txt'}

        gwsnr = GWSNR(**config)

        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        # Validate that output has correct structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

        ###########################################
        # Test custom PSDs for standard detectors #
        ###########################################
        custom_psds = {
            'L1': 'aLIGOaLIGODesignSensitivityT1800044',
            'H1': 'aLIGOaLIGODesignSensitivityT1800044', 
            'V1': 'AdvVirgo'
        }
        
        # Configure GWSNR with reduced verbosity for cleaner test output
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False  # Suppress log messages during error testing
        config['psds'] = custom_psds

        gwsnr = GWSNR(**config)
        snr_result = gwsnr.optimal_snr(gw_param_dict=param_dict)
        # Validate that output has correct structure and numerical properties
        self._validate_snr_output(snr_result, (nsamples,), gwsnr.detector_list)

    def test_multiprocessing_performance(self):
        """
        Tests
        -----
        - SNR calculation with different multiprocessing settings
            - Serial execution (npool=1)
            - Parallel execution with progress bar (imap mode)
            - Parallel execution without progress bar (map mode)
        - Consistency between serial and parallel results
        - Basic performance timing (comparing with interpolation backends)
        """
        # Generate test parameters for BBH events with aligned spins
        nsamples = 1000  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=True     # Include precessing spins
        )

        # Configure GWSNR with reduced verbosity for cleaner test output
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False  # Suppress log messages during error testing
        
        # Test serial execution (no multiprocessing overhead)
        config['npool'] = 1
        gwsnr_serial = GWSNR(**config)
        start_time = time.time()
        serial_snr = gwsnr_serial.optimal_snr(gw_param_dict=param_dict)
        serial_time = time.time() - start_time
        
        # Test parallel with progress bar (imap mode)
        config['npool'] = 4
        config['multiprocessing_verbose'] = True
        gwsnr_parallel_imap = GWSNR(**config)
        start_time = time.time()
        parallel_imap_snr = gwsnr_parallel_imap.optimal_snr(gw_param_dict=param_dict)
        parallel_time_imap = time.time() - start_time

        # Test parallel without progress bar (map mode) 
        config['npool'] = 4
        config['multiprocessing_verbose'] = False
        gwsnr_parallel_map = GWSNR(**config)
        start_time = time.time()
        parallel_map_snr = gwsnr_parallel_map.optimal_snr(gw_param_dict=param_dict)
        parallel_time_map = time.time() - start_time
        
        # Cross-validation between methods
        np.testing.assert_allclose(
            serial_snr["snr_net"],
            parallel_imap_snr["snr_net"],
            rtol=1e-8, err_msg="Serial and parallel_imap should match"
        )
        np.testing.assert_allclose(
            serial_snr["snr_net"],
            parallel_map_snr["snr_net"],
            rtol=1e-8, err_msg="Serial and parallel_map should match"
        )
        
        # Basic timing validation
        assert serial_time > 0 and parallel_time_imap > 0 and parallel_time_map > 0, "Both should take measurable time"

        # Compare with interpolation backend for rough performance benchmark
        config_interp = CONFIG.copy()
        config_interp['gwsnr_verbose'] = False
        config_interp['snr_method'] = "interpolation_aligned_spins"
        config_interp['npool'] = 4

        gwsnr_interp = GWSNR(**config_interp)
        interp_snr = gwsnr_interp.optimal_snr(gw_param_dict=param_dict) # Warm-up call to JIT compile if needed
        start_time = time.time()
        interp_snr = gwsnr_interp.optimal_snr(gw_param_dict=param_dict)
        interp_time = time.time() - start_time

        # Interpolation should be significantly faster than inner product
        print(f"\nTiming (n={nsamples}): serial={serial_time:.3f}s, parallel_imap={parallel_time_imap:.3f}s, parallel_map={parallel_time_map:.3f}s, interpolation={interp_time:.3f}s")
        assert interp_time < serial_time, "Interpolation should be faster than serial inner product"
        assert interp_time < parallel_time_imap, "Interpolation should be faster than parallel_imap"
        assert interp_time < parallel_time_map, "Interpolation should be faster than parallel_map"
        assert parallel_time_imap < serial_time, "Parallel_imap should be faster than serial"
        assert parallel_time_map < serial_time, "Parallel_map should be faster than serial"
