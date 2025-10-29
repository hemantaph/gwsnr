"""
Test suite for validating the GWSNR Artificial Neural Network (ANN) model
for signal-to-noise ratio (SNR) and detection flag (`pdet_net`) calculation
in binary black hole (BBH) systems. These tests assess the consistency,
physical validity, and reproducibility of the ANN model, and benchmark its
detection outputs against the direct inner product method.

Requirements:
-------------
- pip install gwsnr scikit-learn tensorflow
- Upgrading `ml-dtypes` may be required for compatibility.
  - pip install --upgrade ml-dtypes
- pip install pytest

Test Coverage:
--------------
- Spinning BBH systems with spin-precession
- Output validation: dictionary structure, data types, shapes, numerical properties
- Probability of detection (Pdet) consistency between ANN and bilby methods
- Performance: ANN should be faster than full bilby recalculation

Usage:
-----
pytest tests/unit/test_GWSNR_ann.py -v -s
pytest tests/unit/test_GWSNR_ann.py::TestGWSNRANN::test_spinning_bbh -v -s
"""

import os
import numpy as np
import time
from gwsnr import GWSNR
from unit_utils import CommonTestUtils

np.random.seed(1234)

# Default GWSNR configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
DEFAULT_CONFIG = {
    # Computational settings
    'npool': 4,                              # Number of parallel processes for multiprocessing
    
    # Mass parameter ranges for interpolation grid
    'mtot_min': 2*4.98,                      # Minimum total mass (M☉) - typical for BBH
    'mtot_max': 2*112.5+10.0,                # Maximum total mass (M☉) - extended BBH range
    'ratio_min': 0.1,                        # Minimum mass ratio q = m2/m1
    'ratio_max': 1.0,                        # Maximum mass ratio (equal mass)
    'spin_max': 0.99,                        # Maximum dimensionless spin magnitude
    
    # Interpolation grid resolution
    'mtot_resolution': 200,                  # Number of total mass grid points
    'ratio_resolution': 20,                  # Number of mass ratio grid points  
    'spin_resolution': 10,                   # Number of spin grid points
    
    # Waveform generation parameters
    'sampling_frequency': 2048.0,            # Sampling frequency (Hz)
    'waveform_approximant': "IMRPhenomXPHM",    # Waveform model for BBH systems
    'frequency_domain_source_model': 'lal_binary_black_hole',  # LAL source model
    'minimum_frequency': 20.0,               # Low-frequency cutoff (Hz)
    
    # SNR calculation method and settings  
    'snr_method': "ann",  # Use ANN for SNR and pdet calculatio=12]
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

class TestGWSNRANN(CommonTestUtils):
    """
    Test for Artificial Neural Network (ANN) model for SNR and Pdet calculation.
    """

    def test_spinning_bbh(self):
        """
        Tests
        ----------
        - Spinning BBH systems with spin-precession
        """
        # Create configuration for this test (use existing interpolators for speed)
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_pickle')
        config['interpolator_dir'] = gwsnr_dir
        config['snr_method'] = 'ann'
        config['pdet_kwargs'] = dict(snr_th=10.0, snr_th_net=10.0, pdet_type='boolean', distribution_type='noncentral_chi2')
        
        # hybrid initialization
        gwsnr_ann = GWSNR(**config)

        # bilby initialization
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
        nsamples_warm_up = 10
        warm_up_params = self._generate_params(
            nsamples_warm_up, 
            event_type='bbh',        # Binary black hole events
            spin_zero=False,         # Include aligned spin parameters
            spin_precession=True    # Include precessing spins
        )
        # Warm up JIT compilation
        _ = gwsnr_ann.pdet(gw_param_dict=warm_up_params)

        # Test hybrid vs bilby accuracy
        # here pdet is calculated instead of SNR
        times = {}
        start = time.time()
        ann_pdet = gwsnr_ann.pdet(gw_param_dict=param_dict)
        times["hybrid"] = time.time() - start

        start = time.time()
        bilby_pdet = gwsnr_bilby.pdet(gw_param_dict=param_dict)
        times["bilby"] = time.time() - start

        self._validate_pdet_output(ann_pdet, (nsamples,), gwsnr_ann.detector_list, pdet_type='boolean')
        # self._validate_snr_output(bilby_snr, nsamples) # Bilby output already validated in other tests

        # Test Pdet consistency
        agreement = np.mean(np.asarray(ann_pdet['pdet_net']) == np.asarray(bilby_pdet['pdet_net']))
        assert agreement >= 0.9, f"Pdet agreement {agreement:.1%} < 90%"

        # Validate timing relationships
        # self._check_timing_relationships(times["hybrid"], times["bilby"])
        print(f"\nTiming (n={nsamples}): hybrid={times['hybrid']:.3f}s, bilby={times['bilby']:.3f}s")
        assert times["hybrid"] < times["bilby"], "Hybrid should be faster than bilby"
