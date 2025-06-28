:py:mod:`gwsnr.jax.jaxjit_functions`
====================================

.. py:module:: gwsnr.jax.jaxjit_functions

.. autoapi-nested-parse::

   JAX-JIT compiled functions for gravitational wave data analysis.

   This module provides high-performance JAX implementations of core functions used in
   gravitational wave signal-to-noise ratio (SNR) calculations and parameter estimation.
   Key features include:

   - Chirp time calculations using 3.5 post-Newtonian approximations
   - Antenna response pattern computations for gravitational wave detectors
   - Polarization tensor calculations for plus and cross modes
   - Coordinate transformations between celestial and detector frames
   - Vectorized operations for efficient batch processing
   - Automatic parallelization through JAX's vmap for multi-dimensional arrays

   All functions are compiled with JAX's @jit decorator for optimal performance,
   automatic differentiation support, and GPU acceleration. The implementations
   are optimized for use in Bayesian inference pipelines and matched filtering
   applications in gravitational wave astronomy.

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.jax.jaxjit_functions.findchirp_chirptime_jax
   gwsnr.jax.jaxjit_functions.einsum1
   gwsnr.jax.jaxjit_functions.einsum2
   gwsnr.jax.jaxjit_functions.gps_to_gmst
   gwsnr.jax.jaxjit_functions.ra_dec_to_theta_phi
   gwsnr.jax.jaxjit_functions.get_polarization_tensor_plus
   gwsnr.jax.jaxjit_functions.get_polarization_tensor_cross
   gwsnr.jax.jaxjit_functions.antenna_response_plus
   gwsnr.jax.jaxjit_functions.antenna_response_cross
   gwsnr.jax.jaxjit_functions.antenna_response_array



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

.. py:function:: einsum1(m, n)

   
   Function to calculate the outer product of two 3D vectors (JAX implementation).


   :Parameters:

       **m** : `jax.numpy.ndarray`
           3D vector (length 3).

       **n** : `jax.numpy.ndarray`
           3D vector (length 3).

   :Returns:

       **outer_product** : `jax.numpy.ndarray`
           3x3 matrix representing the outer product of m and n.








   .. rubric:: Notes

   JAX implementation uses jnp.outer for efficient computation with automatic
   differentiation and GPU acceleration support. Equivalent to the tensor
   product m ⊗ n used in gravitational wave polarization calculations.





   ..
       !! processed by numpydoc !!

.. py:function:: einsum2(m, n)

   
   Function to calculate the Frobenius inner product of two 3x3 matrices (JAX implementation).


   :Parameters:

       **m** : `jax.numpy.ndarray`
           3x3 matrix.

       **n** : `jax.numpy.ndarray`
           3x3 matrix.

   :Returns:

       **inner_product** : `float`
           Scalar result of the element-wise multiplication and sum of the two matrices.








   .. rubric:: Notes

   Computes the trace of the element-wise product of two matrices, equivalent to
   the Frobenius inner product. Used in antenna response calculations for
   gravitational wave detectors. JAX implementation leverages vectorized operations
   for efficient computation with automatic differentiation support.





   ..
       !! processed by numpydoc !!

.. py:function:: gps_to_gmst(gps_time)

   
   Function to convert GPS time to Greenwich Mean Sidereal Time (GMST) (JAX implementation).


   :Parameters:

       **gps_time** : `float`
           GPS time in seconds.

   :Returns:

       **gmst** : `float`
           Greenwich Mean Sidereal Time in radians.








   .. rubric:: Notes

   Uses a linear approximation with a reference time and slope to compute GMST.
   The reference time (time0) is 1126259642.413 seconds and the slope is
   7.292115855382993e-05 radians per second, which approximates Earth's rotation rate.
   JAX implementation supports automatic differentiation for gradient-based optimization.





   ..
       !! processed by numpydoc !!

.. py:function:: ra_dec_to_theta_phi(ra, dec, gmst)

   
   Function to convert right ascension and declination to spherical coordinates (JAX implementation).


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **gmst** : `float`
           Greenwich Mean Sidereal Time in radians.

   :Returns:

       **theta** : `float`
           Polar angle (colatitude) in radians, measured from the north pole.

       **phi** : `float`
           Azimuthal angle in radians, adjusted for Earth's rotation.








   .. rubric:: Notes

   Converts celestial coordinates (ra, dec) to spherical coordinates (theta, phi)
   in the detector frame. The azimuthal angle is corrected for Earth's rotation
   using GMST. Theta represents the angle from the north pole (colatitude).
   JAX implementation provides automatic differentiation capabilities for
   parameter estimation and optimization workflows.





   ..
       !! processed by numpydoc !!

.. py:function:: get_polarization_tensor_plus(ra, dec, time, psi)

   
   Function to calculate the plus polarization tensor for gravitational wave detection (JAX implementation).


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source in seconds.

       **psi** : `float`
           Polarization angle of the source in radians.

   :Returns:

       **polarization_tensor_plus** : `jax.numpy.ndarray`
           3x3 plus polarization tensor matrix (m⊗m - n⊗n).








   .. rubric:: Notes

   Calculates the plus polarization tensor in the detector frame by first converting
   celestial coordinates to spherical coordinates using GMST, then computing
   the basis vectors m and n based on the polarization angle psi. Returns the
   tensor m⊗m - n⊗n for plus polarization mode. JAX implementation supports
   automatic differentiation and GPU acceleration for efficient computation.





   ..
       !! processed by numpydoc !!

.. py:function:: get_polarization_tensor_cross(ra, dec, time, psi)

   
   Function to calculate the cross polarization tensor for gravitational wave detection (JAX implementation).


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source in seconds.

       **psi** : `float`
           Polarization angle of the source in radians.

   :Returns:

       **polarization_tensor_cross** : `jax.numpy.ndarray`
           3x3 cross polarization tensor matrix (m⊗n + n⊗m).








   .. rubric:: Notes

   Calculates the cross polarization tensor in the detector frame by first converting
   celestial coordinates to spherical coordinates using GMST, then computing
   the basis vectors m and n based on the polarization angle psi. Returns the
   tensor m⊗n + n⊗m for cross polarization mode. JAX implementation supports
   automatic differentiation and GPU acceleration for efficient computation.





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

