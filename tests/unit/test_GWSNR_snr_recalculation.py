"""
Unit tests for GWSNR hybrid SNR recalculation functionality.

Tests the hybrid approach: fast interpolation + selective bilby recalculation 
for events within [4, 12] SNR range.

Test Coverage:
--------------
- Hybrid vs bilby SNR accuracy for spin-precessing BBH, using IMRPhenomXPHM
- Output validation and Pdet consistency  
- Performance comparison (hybrid should be faster)

Usage:
-----
Run individual tests: pytest tests/unit/test_GWSNR_snr_recalculation.py::TestGWSNRSNRRecalculation::test_name
Run full suite: pytest tests/unit/test_GWSNR_snr_recalculation.py -v -s
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
    'snr_method': "interpolation_aligned_spins",  # Use interpolation with aligned spins
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
    'pdet': False,                           # Calculate SNR, not probability of detection
    'snr_th': 8.0,                          # Single-detector SNR threshold
    'snr_th_net': 8.0,                      # Network SNR threshold

    # SNR recalculation settings
    'snr_recalculation': True,
    'snr_recalculation_range': [4, 12],
    'snr_recalculation_waveform_approximant': "IMRPhenomXPHM",
}

class TestGWSNRSNRRecalculation(CommonTestUtils):
    """Test hybrid SNR recalculation: interpolation + selective bilby recalculation."""

    def test_spinning_bbh(self):
        """
        Tests
        -----
        - Hybrid SNR recalculation accuracy vs full bilby for spinning BBH events
        - Output validation: dictionary structure, data types, shapes, numerical properties
        - Probability of detection (Pdet) consistency between hybrid and bilby methods
        - Performance: hybrid should be faster than full bilby recalculation
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = CONFIG.copy()
        config['gwsnr_verbose'] = False
        
        # hybrid initialization
        gwsnr_hybrid = GWSNR(**config)

        # bilby initialization
        config['snr_recalculation'] = False
        config['snr_method'] = 'inner_product'
        gwsnr_bilby = GWSNR(**config)

        # Generate test parameters for BBH events with aligned spins
        nsamples = 5000  # Number of test events
        param_dict = self._generate_params(
            nsamples, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=True    # Include precessing spins
        )

        # Test hybrid vs bilby accuracy
        times = {}
        hybrid_snr = gwsnr_hybrid.snr_with_interpolation(gw_param_dict=param_dict)  # Warm-up call to JIT compile
        start = time.time()
        hybrid_snr = gwsnr_hybrid.snr(gw_param_dict=param_dict)
        times["hybrid"] = time.time() - start

        start = time.time()
        bilby_snr = gwsnr_bilby.snr(gw_param_dict=param_dict)
        times["bilby"] = time.time() - start

        self._validate_output(hybrid_snr, (nsamples,), gwsnr_hybrid.detector_list, pdet=False)
        # self._validate_snr_output(bilby_snr, nsamples) # Bilby output already validated in other tests

        # Verify recalculated events match bilby
        hybrid_arr = np.asarray(hybrid_snr["snr_net"])
        bilby_arr = np.asarray(bilby_snr["snr_net"])

        recalc_mask = (hybrid_arr >= 4) & (hybrid_arr <= 12)
        recalc_indices = np.where(recalc_mask)[0]

        if len(recalc_indices) > 0:
            assert np.allclose(hybrid_arr[recalc_indices], bilby_arr[recalc_indices], rtol=1e-6)

        # Test Pdet consistency
        pdet_hybrid = gwsnr_hybrid.probability_of_detection(snr_dict=hybrid_snr, type="bool")["pdet_net"]
        pdet_bilby = gwsnr_bilby.probability_of_detection(snr_dict=bilby_snr, type="bool")["pdet_net"]

        agreement = np.mean(np.asarray(pdet_hybrid) == np.asarray(pdet_bilby))
        assert agreement >= 0.9, f"Pdet agreement {agreement:.1%} < 90%"
        
        # Validate timing relationships
        # self._check_timing_relationships(times["hybrid"], times["bilby"])
        print(f"\nTiming (n={nsamples}): hybrid={times['hybrid']:.3f}s, bilby={times['bilby']:.3f}s")
        assert times["hybrid"] < times["bilby"], "Hybrid should be faster than bilby"
        
