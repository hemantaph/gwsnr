"""
Unit Test Utilities for GWSNR Testing Suite

This module provides common utilities and helper functions for the GWSNR unit testing
framework using pytest. Pytest is a powerful testing framework that makes it easy to 
write simple tests and scales to complex functional testing.

Test Coverage:
- Parameter generation for various gravitational wave events (BBH, BNS, NSBH)
- Output validation for SNR calculations and probability of detection
- Support for different spin configurations (no spins, aligned spins, precessing spins)
- Numerical validation ensuring finite, real, non-negative values
"""
import numpy as np

np.random.seed(1234)

class CommonTestUtils:
    """Shared utilities for GWSNR unit tests."""

    def _generate_params(self, nsamples, event_type='bbh', spin_zero=True, spin_precession=False):
        """Generate random gravitational wave parameters for testing.
        
        Parameters
        ----------
            nsamples : `int`
                Number of sample events to generate
            event_type : `str`
                Type of event ('bbh', 'bns', 'nsbh')
            spin_zero : `bool`
                If True, excludes spin parameters
            spin_precession : `bool`
                If True, includes precessing spin parameters

        Returns
        -------
        param_dict : `dict`
            Dictionary of parameters compatible with GWSNR input format
        """
        # Define mass and distance ranges for different compact binary types
        # BBH: Binary Black Hole, BNS: Binary Neutron Star, NSBH: Neutron Star-Black Hole
        event_params = {
            'bbh': {'mtot_min': 2*4.98, 'mtot_max': 2*112.5, 'distance': 500.0},  # Typical BBH masses and distance
            'bns': {'mtot_min': 2*1.0, 'mtot_max': 2*3.0, 'distance': 100.0},    # Typical BNS masses and distance
            'nsbh': {'mtot_min': 2*3.0, 'mtot_max': 2*10.0, 'distance': 300.0}   # Typical NSBH masses and distance
        }
        
        # Validate event type input
        if event_type not in event_params:
            raise ValueError(f"Unsupported event_type: {event_type}. Use 'bbh', 'bns', or 'nsbh'.")
            
        # Extract parameters for the specified event type
        params = event_params[event_type]
        mtot_min, mtot_max = params['mtot_min'], params['mtot_max']  # Total mass range in solar masses
        luminosity_distance = params['distance']  # Luminosity distance in Mpc
        
        # Generate random mass parameters
        mtot = np.random.uniform(mtot_min, mtot_max, nsamples)  # Total mass (M☉)
        mass_ratio = np.random.uniform(0.2, 1, nsamples)        # Mass ratio q = m2/m1, with q ≤ 1
        
        # Build parameter dictionary with physical and sky location parameters
        param_dict = {
            # Binary mass parameters (convert from total mass and ratio to component masses)
            'mass_1': mtot / (1 + mass_ratio),                      # Primary mass (M☉)
            'mass_2': mtot * mass_ratio / (1 + mass_ratio),         # Secondary mass (M☉)
            
            # Distance parameter
            'luminosity_distance': luminosity_distance * np.ones(nsamples),  # Distance (Mpc)
            
            # Binary orientation parameters
            'theta_jn': np.random.uniform(0, 2*np.pi, nsamples),    # Inclination angle (rad)
            
            # Sky location parameters  
            'ra': np.random.uniform(0, 2*np.pi, nsamples),          # Right ascension (rad)
            'dec': np.random.uniform(-np.pi/2, np.pi/2, nsamples),  # Declination (rad)
            'psi': np.random.uniform(0, 2*np.pi, nsamples),         # Polarization angle (rad)
            
            # Signal parameters
            'phase': np.random.uniform(0, 2*np.pi, nsamples),       # Coalescence phase (rad)
            'geocent_time': 1246527224.169434 * np.ones(nsamples), # GPS time (s) - fixed for consistency
        }

        # Add spin parameters based on requested configuration
        if not spin_zero:
            # Add aligned spin parameters (spins aligned/anti-aligned with orbital angular momentum)
            param_dict.update({
                'a_1': np.random.uniform(-0.8, 0.8, nsamples),  # Primary dimensionless spin (-1 to 1)
                'a_2': np.random.uniform(-0.8, 0.8, nsamples)   # Secondary dimensionless spin (-1 to 1)
            })

        if spin_precession:
            # Add precessing spin parameters (spins can point in any direction)
            param_dict.update({
                'a_1': np.random.uniform(0, 0.8, nsamples),        # Primary spin magnitude (0 to 1)
                'a_2': np.random.uniform(0, 0.8, nsamples),        # Secondary spin magnitude (0 to 1)
                'tilt_1': np.random.uniform(0, np.pi, nsamples),   # Primary spin tilt angle (rad)
                'tilt_2': np.random.uniform(0, np.pi, nsamples),   # Secondary spin tilt angle (rad)  
                'phi_12': np.random.uniform(0, 2*np.pi, nsamples), # Azimuthal angle between spins (rad)
                'phi_jl': np.random.uniform(0, 2*np.pi, nsamples)  # Azimuthal angle between J and L (rad)
            })

        return param_dict

    def _validate_output(self, snr_dict, expected_shape, detector_list, pdet=False):
        """Validate SNR output structure and numerical properties.
        
        Parameters
        ----------
            snr_dict : `dict`
                Dictionary containing SNR or probability detection values
            expected_shape: `tuple`
                Expected shape of output arrays
            detector_list: `list`
                List of detector names to validate
            pdet: `Union[bool, str]`
                Type of probability detection ('bool', 'matched_filter', or False for SNR)
        """
        # Validate that output is a dictionary
        assert isinstance(snr_dict, dict), "Output must be a dictionary"
        
        # Create list of all keys to validate (individual detectors + network)
        test_keys = detector_list.copy()  # Start with individual detector names
        if pdet is False:
            test_keys.append("snr_net")  # Add network SNR key for regular SNR calculations
        else:
            test_keys.append("pdet_net")         # Add network pdet key for probability calculations

        # Validate each detector and network output
        for key in test_keys:
            # Check that required key exists in output dictionary
            assert key in snr_dict, f"Missing {key} in output"
            values = snr_dict[key]
            
            # Validate array shape matches expected dimensions
            assert values.shape == expected_shape, \
                f"Shape mismatch for {key}: expected {expected_shape}, got {values.shape}"
            
            # Apply validation rules based on the type of output
            if pdet == 'bool':
                # Boolean probability of detection: values must be 0 (not detected) or 1 (detected)
                assert np.all(np.isin(values, [0, 1])), f"{key} values must be binary (0 or 1)"
                
            elif pdet == 'matched_filter':
                # Matched-filter probability: continuous values between 0 and 1
                assert np.all((values >= 0) & (values <= 1)), f"{key} values must be in [0, 1]"
                
            elif pdet is False:
                # Regular SNR values: must be positive real numbers
                assert values.dtype == np.float64, f"{key} expected float64, got {values.dtype}"
                assert np.all(np.isfinite(values)), f"{key} values must be finite (no NaN/inf)"
                assert np.all(np.isreal(values)), f"{key} values must be real (no complex numbers)"
                assert np.all(values >= 0), f"{key} values must be non-negative (SNR ≥ 0)"
