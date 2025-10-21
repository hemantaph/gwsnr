:py:mod:`gwsnr.jax`
===================

.. py:module:: gwsnr.jax


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   jaxjit_functions/index.rst
   jaxjit_interpolators/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.jax.findchirp_chirptime_jax
   gwsnr.jax.antenna_response_plus
   gwsnr.jax.antenna_response_cross
   gwsnr.jax.antenna_response_array
   gwsnr.jax.get_interpolated_snr_aligned_spins_jax
   gwsnr.jax.get_interpolated_snr_no_spins_jax



.. py:function:: findchirp_chirptime_jax(m1, m2, fmin)

   
   Function to calculate the chirp time from minimum frequency to last stable orbit (JAX implementation).

   Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.

   :Parameters:

       **m1** : `float`
           Mass of the first body in solar masses.

       **m2** : `float`
           Mass of the second body in solar masses.

       **fmin** : `float`
           Lower frequency cutoff in Hz.

   :Returns:

       **chirp_time** : `float`
           Time taken from f_min to f_lso (last stable orbit frequency) in seconds.








   .. rubric:: Notes

   Calculates chirp time using 3.5PN approximation for gravitational wave Fourier phase.
   The time represents frequency evolution from fmin to last stable orbit frequency.
   Uses post-Newtonian expansion coefficients optimized for efficient JAX computation.
   JAX implementation supports automatic differentiation and GPU acceleration.





   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_plus(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the plus polarization antenna response for gravitational wave detection (JAX implementation).


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source in seconds.

       **psi** : `float`
           Polarization angle of the source in radians.

       **detector_tensor** : `jax.numpy.ndarray`
           Detector tensor for the detector (3x3 matrix).

   :Returns:

       **antenna_response_plus** : `float`
           Plus polarization antenna response of the detector.








   .. rubric:: Notes

   Computes the plus polarization antenna response by calculating the Frobenius
   inner product between the detector tensor and the plus polarization tensor.
   The polarization tensor is determined by the source location (ra, dec),
   observation time, and polarization angle (psi). JAX implementation provides
   automatic differentiation for parameter estimation workflows.





   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_cross(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the cross polarization antenna response for gravitational wave detection (JAX implementation).


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source in seconds.

       **psi** : `float`
           Polarization angle of the source in radians.

       **detector_tensor** : `jax.numpy.ndarray`
           Detector tensor for the detector (3x3 matrix).

   :Returns:

       **antenna_response_cross** : `float`
           Cross polarization antenna response of the detector.








   .. rubric:: Notes

   Computes the cross polarization antenna response by calculating the Frobenius
   inner product between the detector tensor and the cross polarization tensor.
   The polarization tensor is determined by the source location (ra, dec),
   observation time, and polarization angle (psi). JAX implementation provides
   automatic differentiation for parameter estimation workflows.





   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_array(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response for multiple detectors and sources (JAX implementation).


   :Parameters:

       **ra** : `jax.numpy.ndarray`
           Array of right ascension values for sources in radians.

       **dec** : `jax.numpy.ndarray`
           Array of declination values for sources in radians.

       **time** : `jax.numpy.ndarray`
           Array of GPS times for sources in seconds.

       **psi** : `jax.numpy.ndarray`
           Array of polarization angles for sources in radians.

       **detector_tensor** : `jax.numpy.ndarray`
           Detector tensor array for multiple detectors (n×3×3 matrix), where n is the number of detectors.

   :Returns:

       **Fp** : `jax.numpy.ndarray`
           Plus polarization antenna response array with shape (n_detectors, n_sources).

       **Fc** : `jax.numpy.ndarray`
           Cross polarization antenna response array with shape (n_detectors, n_sources).








   .. rubric:: Notes

   Computes antenna responses for both plus and cross polarizations across multiple
   detectors and source parameters simultaneously. Uses JAX's vmap for efficient
   vectorized computation with automatic differentiation support. Each antenna
   response is calculated using the Frobenius inner product between detector
   tensors and polarization tensors derived from source sky location and
   polarization angle. Optimized for GPU acceleration and gradient-based optimization.





   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_aligned_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000)

   
   Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals with aligned spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.


   :Parameters:

       **mass_1** : jax.numpy.ndarray
           Primary mass of the binary system in solar masses.

       **mass_2** : jax.numpy.ndarray
           Secondary mass of the binary system in solar masses.

       **luminosity_distance** : jax.numpy.ndarray
           Luminosity distance to the source in Mpc.

       **theta_jn** : jax.numpy.ndarray
           Inclination angle between the orbital angular momentum and line of sight in radians.

       **psi** : jax.numpy.ndarray
           Polarization angle in radians.

       **geocent_time** : jax.numpy.ndarray
           GPS time of coalescence at the geocenter in seconds.

       **ra** : jax.numpy.ndarray
           Right ascension of the source in radians.

       **dec** : jax.numpy.ndarray
           Declination of the source in radians.

       **a_1** : jax.numpy.ndarray
           Dimensionless spin magnitude of the primary black hole.

       **a_2** : jax.numpy.ndarray
           Dimensionless spin magnitude of the secondary black hole.

       **detector_tensor** : jax.numpy.ndarray
           Detector tensor array containing detector response information.
           Shape: (n_detectors, ...)

       **snr_partialscaled** : jax.numpy.ndarray
           Pre-computed scaled partial SNR values for interpolation.
           Shape: (n_detectors, ...)

       **ratio_arr** : jax.numpy.ndarray
           Mass ratio grid points for interpolation (q = m2/m1).

       **mtot_arr** : jax.numpy.ndarray
           Total mass grid points for interpolation.

       **a1_arr** : jax.numpy.ndarray
           Primary spin grid points for interpolation.

       **a_2_arr** : jax.numpy.ndarray
           Secondary spin grid points for interpolation.

   :Returns:

       **snr** : jax.numpy.ndarray
           SNR values for each detector. Shape: (n_detectors, n_samples)

       **snr_effective** : jax.numpy.ndarray
           Effective SNR combining all detectors. Shape: (n_samples,)

       **snr_partial_** : jax.numpy.ndarray
           Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)

       **d_eff** : jax.numpy.ndarray
           Effective distance for each detector accounting for antenna response.
           Shape: (n_detectors, n_samples)








   .. rubric:: Notes

   - Uses 4D cubic spline interpolation for efficient SNR calculation
   - Assumes aligned spins (no precession)
   - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
   - Chirp mass and inclination-dependent factors are computed analytically





   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_no_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000)

   
   Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals with aligned spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.


   :Parameters:

       **mass_1** : jax.numpy.ndarray
           Primary mass of the binary system in solar masses.

       **mass_2** : jax.numpy.ndarray
           Secondary mass of the binary system in solar masses.

       **luminosity_distance** : jax.numpy.ndarray
           Luminosity distance to the source in Mpc.

       **theta_jn** : jax.numpy.ndarray
           Inclination angle between the orbital angular momentum and line of sight in radians.

       **psi** : jax.numpy.ndarray
           Polarization angle in radians.

       **geocent_time** : jax.numpy.ndarray
           GPS time of coalescence at the geocenter in seconds.

       **ra** : jax.numpy.ndarray
           Right ascension of the source in radians.

       **dec** : jax.numpy.ndarray
           Declination of the source in radians.

       **a_1** : jax.numpy.ndarray
           Dimensionless spin magnitude of the primary black hole.

       **a_2** : jax.numpy.ndarray
           Dimensionless spin magnitude of the secondary black hole.

       **detector_tensor** : jax.numpy.ndarray
           Detector tensor array containing detector response information.
           Shape: (n_detectors, ...)

       **snr_partialscaled** : jax.numpy.ndarray
           Pre-computed scaled partial SNR values for interpolation.
           Shape: (n_detectors, ...)

       **ratio_arr** : jax.numpy.ndarray
           Mass ratio grid points for interpolation (q = m2/m1).

       **mtot_arr** : jax.numpy.ndarray
           Total mass grid points for interpolation.

       **a1_arr** : jax.numpy.ndarray
           Primary spin grid points for interpolation.

       **a_2_arr** : jax.numpy.ndarray
           Secondary spin grid points for interpolation.

   :Returns:

       **snr** : jax.numpy.ndarray
           SNR values for each detector. Shape: (n_detectors, n_samples)

       **snr_effective** : jax.numpy.ndarray
           Effective SNR combining all detectors. Shape: (n_samples,)

       **snr_partial_** : jax.numpy.ndarray
           Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)

       **d_eff** : jax.numpy.ndarray
           Effective distance for each detector accounting for antenna response.
           Shape: (n_detectors, n_samples)








   .. rubric:: Notes

   - Uses 4D cubic spline interpolation for efficient SNR calculation
   - Assumes aligned spins (no precession)
   - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
   - Chirp mass and inclination-dependent factors are computed analytically





   ..
       !! processed by numpydoc !!

