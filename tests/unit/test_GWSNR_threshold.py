"""
Unit Tests for SNRThresholdFinder (Threshold optimization via cross-entropy method)

This test suite validates the SNR threshold optimization functionality using
cross-entropy methods for gravitational wave detection statistics.

Requirements:
-------------
- pip install gwsnr
- pip install pytest

Test Coverage:
--------------
- Threshold optimization with real astrophysical BBH injection data
- Cross-entropy method convergence and numerical stability
- Output validation: data types, bounds, and numerical properties
- Performance benchmarking for threshold finding algorithms
- Error handling for invalid configurations and missing parameters

Usage:
-----
pytest tests/unit/test_GWSNR_threshold.py -v -s
pytest tests/unit/test_GWSNR_threshold.py::TestSNRThresholdFinder::test_name -v -s
"""

import os
import time
import tempfile
import numpy as np
from gwsnr.utils import load_json
from gwsnr.threshold import SNRThresholdFinder

np.random.seed(1234)

# Default SNRThresholdFinder configuration dictionary for all tests
# This provides a consistent baseline that individual tests can modify as needed
DEFAULT_CONFIG = {
    # Computational settings
    'npool': 4,                              # Number of parallel processes for multiprocessing
    
    # Selection range parameters for mass filtering
    'selection_range': {
        'key_name': 'mass1_source',          # Primary mass parameter for filtering
        'parameter': None,                   # Will be populated with actual data
        'range': (30, 60),                   # Mass range in solar masses for BBH
    },
    
    # Original detection statistic configuration (e.g., GSTLAL FAR)
    'original_detection_statistic': {
        'key_name': 'gstlal_far',            # False alarm rate from GSTLAL pipeline
        'parameter': None,                   # Will be populated with actual data
        'threshold': 1,                      # 1 detection per year threshold
    },
    
    # Projected detection statistic configuration (network SNR)
    'projected_detection_statistic': {
        'key_name': 'observed_snr_net',      # Observed network SNR
        'parameter': None,                   # Will be populated with actual data
        'threshold': None,                   # To be optimized by cross-entropy method
        'threshold_search_bounds': (6, 12),  # Search range for optimal SNR threshold
    },
    
    # Parameters for fitting (redshift distribution)
    'parameters_to_fit': {
        'key_name': ['z'],                   # Redshift parameter for rate calculations
        'parameter': None,                   # Will be populated with actual data
    },
    
    # Analysis settings
    'sample_size': 20000,                    # Sample size for cross-entropy optimization
    'multiprocessing_verbose': True,         # Enable detailed multiprocessing logs
}

class TestSNRThresholdFinder:
    """
    Test suite for SNR threshold optimization using cross-entropy methods.
    """

    def test_find_best_threshold_known_shape(self):
        """
        Tests
        -----
        - Cross-entropy optimization for SNR threshold determination
        - Convergence and numerical stability of threshold finding algorithm
        - Output validation: data types, bounds, and numerical properties
        - Performance benchmarking for real astrophysical BBH injection data
        - Integration with O4 injection catalog parameters (mass, redshift, FAR, SNR)
        """
        
        # Load astrophysical BBH parameters from Reed Essick O4 injection catalog
        # Data source: Zenodo repository with necessary parameters extracted
        test_dir = os.path.dirname(__file__)
        injection_data = os.path.join(test_dir, 'injection_data.json')
        gw_params = load_json(injection_data)  # 20,000 BBH injections

        # Create configuration for this test with real injection data
        config = DEFAULT_CONFIG.copy()
        gwsnr_dir = os.path.dirname(__file__)
        gwsnr_dir = os.path.join(gwsnr_dir, '../interpolator_pickle')
        config['interpolator_dir'] = gwsnr_dir
        config.update({
            'multiprocessing_verbose': True,     # Enable detailed progress logging
            'sample_size': 10000,               # Reduced sample size for faster testing
            
            # Populate configuration with actual injection parameters
            'selection_range': {
                'key_name': 'mass1_source',
                'parameter': np.array(gw_params['mass1_source']),  # Primary mass data
                'range': (30, 60),                                 # BBH mass range (M☉)
            },
            'original_detection_statistic': {
                'key_name': 'gstlal_far', 
                'parameter': np.array(gw_params['gstlal_far']),    # GSTLAL false alarm rates
                'threshold': 1,                                    # 1 per year threshold
            },
            'projected_detection_statistic': {
                'key_name': 'observed_snr_net',
                'parameter': np.array(gw_params['observed_snr_net']), # Network SNR observations
                'threshold': None,                                     # To be optimized
                'threshold_search_bounds': (6, 12),                   # SNR search range
            },
            'parameters_to_fit': {
                'key_name': ['z'],
                'parameter': np.array(gw_params['z']),             # Redshift distribution
            }
        })

        # Execute threshold optimization with performance timing
        start_time = time.time()
        threshold_finder = SNRThresholdFinder(**config)
        
        # Run cross-entropy optimization algorithm
        best_threshold, delta_H, H_values, H_true, snr_thresholds = threshold_finder.find_threshold(
            iteration=6,              # Number of optimization iterations
            print_output=True,        # Enable progress output
            no_multiprocessing=True  # Use parallel processing for speed
        )
        
        execution_time = time.time() - start_time

        # VALIDATION: Data type and numerical properties
        assert isinstance(best_threshold, float), \
            f"Expected float for best_threshold, got {type(best_threshold)}"
        
        assert np.isfinite(best_threshold), \
            f"best_threshold must be finite (no NaN/Inf), got {best_threshold}"
        
        assert np.isreal(best_threshold), \
            f"best_threshold must be real (no complex numbers), got {best_threshold}"

        # VALIDATION: Astrophysically reasonable threshold range
        assert 4 < best_threshold < 15, \
            f"best_threshold should be in reasonable astrophysical range [4, 15], got {best_threshold}"

        # VALIDATION: Performance constraint (should complete within reasonable time)
        assert execution_time < 60, \
            f"Threshold optimization took too long: {execution_time:.2f}s > 60s"
        
        print(f"\nOptimization results:")
        print(f"  - Best SNR threshold: {best_threshold:.3f}")
