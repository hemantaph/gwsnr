:orphan:

:py:mod:`gwsnr.jax`
===================

.. py:module:: gwsnr.jax


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   jaxjit_functions/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.jax.findchirp_chirptime_jax
   gwsnr.jax.einsum1
   gwsnr.jax.einsum2
   gwsnr.jax.gps_to_gmst
   gwsnr.jax.ra_dec_to_theta_phi
   gwsnr.jax.get_polarization_tensor_plus
   gwsnr.jax.get_polarization_tensor_cross
   gwsnr.jax.antenna_response_plus
   gwsnr.jax.antenna_response_cross
   gwsnr.jax.antenna_response_array
   gwsnr.jax.find_index_1d_jax
   gwsnr.jax.cubic_function_4pts_jax
   gwsnr.jax.cubic_spline_4d_jax
   gwsnr.jax.cubic_spline_4d_batched_jax
   gwsnr.jax.get_interpolated_snr_aligned_spins_jax
   gwsnr.jax.cubic_spline_2d_jax
   gwsnr.jax.cubic_spline_2d_batched_jax
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

.. py:function:: find_index_1d_jax(x_array, x_new)

   
   Find the index for cubic spline interpolation in 1D.
   Returns the index and a condition for edge handling.


   :Parameters:

       **x_array** : jnp.ndarray
           The array of x values for interpolation. Must be sorted in ascending order.

       **x_new** : float or jnp.ndarray
           The new x value(s) to find the index for.

   :Returns:

       **i** : jnp.ndarray
           The index in `x_array` where `x_new` would fit, clipped to range [1, N-3]
           where N is the length of x_array.

       **condition_i** : jnp.ndarray
           A condition indicating which interpolation branch to use:
           - 1: Use linear interpolation at the left edge (x_new <= x_array[1]).
           - 2: Use cubic interpolation in the middle.
           - 3: Use linear interpolation at the right edge (x_new >= x_array[N-2]).








   .. rubric:: Notes

   Uses binary search with clipped indices to ensure valid 4-point stencils.
   The condition parameter determines linear vs cubic interpolation at boundaries.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_function_4pts_jax(x_eval, x_pts, y_pts, condition_i)

   
   Performs piecewise interpolation using 4 points with JAX compatibility.
   This function implements a piecewise interpolation scheme that uses:
   - Linear interpolation at the left boundary (condition_i=1)
   - Cubic interpolation in the middle region (condition_i=2)
   - Linear interpolation at the right boundary (condition_i=3)
   The cubic interpolation uses cubic Hermite spline coefficients for smooth
   interpolation between the middle two points, while the boundary regions
   use linear interpolation for stability.
   :param x_eval: The x-coordinate(s) where interpolation is to be evaluated.
   :type x_eval: array_like
   :param x_pts: Array of 4 x-coordinates of the interpolation points, ordered as
                 [x0, x1, x2, x3] where x1 and x2 are the main interpolation interval.
   :type x_pts: array_like
   :param y_pts: Array of 4 y-coordinates corresponding to x_pts, ordered as
                 [y0, y1, y2, y3].
   :type y_pts: array_like
   :param condition_i: Interpolation mode selector:
                       - 1: Linear interpolation using points (x0, y0) and (x1, y1)
                       - 2: Cubic interpolation using all 4 points with x_eval in [x1, x2]
                       - 3: Linear interpolation using points (x2, y2) and (x3, y3)
   :type condition_i: int

   :Returns:

       array_like
           Interpolated value(s) at x_eval using the specified interpolation method.








   .. rubric:: Notes

   - The function handles degenerate cases where denominators are zero by
     returning appropriate fallback values (y0, y1, or y2 respectively).
   - Uses JAX's lax.switch for efficient conditional execution.
   - The cubic interpolation uses normalized parameter t = (x_eval - x1) / (x2 - x1).
   - Cubic coefficients follow the pattern: a*t³ + b*t² + c*t + d where:





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new)

   
   Perform 4D cubic spline interpolation using JAX operations.
   This function interpolates a 4D array (snrpartialscaled_array) at specified points
   using cubic spline interpolation. The interpolation is performed sequentially
   along each dimension: first a2, then a1, then mtot, and finally q.


   :Parameters:

       **q_array** : jax.numpy.ndarray
           1D array containing the q-dimension coordinate values.

       **mtot_array** : jax.numpy.ndarray
           1D array containing the total mass dimension coordinate values.

       **a1_array** : jax.numpy.ndarray
           1D array containing the first spin parameter dimension coordinate values.

       **a2_array** : jax.numpy.ndarray
           1D array containing the second spin parameter dimension coordinate values.

       **snrpartialscaled_array** : jax.numpy.ndarray
           4D array containing the SNR partial scaled values to be interpolated.
           Shape should be (len(q_array), len(mtot_array), len(a1_array), len(a2_array)).

       **q_new** : float
           New q value at which to interpolate.

       **mtot_new** : float
           New total mass value at which to interpolate.

       **a1_new** : float
           New first spin parameter value at which to interpolate.

       **a2_new** : float
           New second spin parameter value at which to interpolate.

   :Returns:

       float
           Interpolated SNR value at the specified (q_new, mtot_new, a1_new, a2_new) point.








   .. rubric:: Notes

   This function uses nested loops to perform interpolation sequentially along each
   dimension. It relies on helper functions `find_index_1d_jax` for finding array
   indices and `cubic_function_4pts_jax` for 1D cubic interpolation using 4 points.
   The interpolation process:
   1. Find indices and interpolation weights for each dimension
   2. Interpolate along a2 dimension for each combination of q, mtot, a1 indices
   3. Interpolate along a1 dimension using results from step 2
   4. Interpolate along mtot dimension using results from step 3
   5. Interpolate along q dimension to get the final result





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_batched_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_batch, mtot_batch, a1_batch, a2_batch)

   
   Perform batched 4D cubic spline interpolation using JAX vectorization.
   This function applies 4D cubic spline interpolation to batches of input parameters
   using JAX's vmap for efficient vectorized computation. It interpolates SNR values
   based on mass ratio (q), total mass (mtot), and two spin parameters (a1, a2).


   :Parameters:

       **q_array** : jax.numpy.ndarray
           1D array of mass ratio grid points for interpolation.

       **mtot_array** : jax.numpy.ndarray
           1D array of total mass grid points for interpolation.

       **a1_array** : jax.numpy.ndarray
           1D array of first spin parameter grid points for interpolation.

       **a2_array** : jax.numpy.ndarray
           1D array of second spin parameter grid points for interpolation.

       **snrpartialscaled_array** : jax.numpy.ndarray
           4D array of SNR values corresponding to the grid points, with shape
           (len(q_array), len(mtot_array), len(a1_array), len(a2_array)).

       **q_batch** : jax.numpy.ndarray
           1D array of mass ratio values to interpolate at.

       **mtot_batch** : jax.numpy.ndarray
           1D array of total mass values to interpolate at.

       **a1_batch** : jax.numpy.ndarray
           1D array of first spin parameter values to interpolate at.

       **a2_batch** : jax.numpy.ndarray
           1D array of second spin parameter values to interpolate at.

   :Returns:

       jax.numpy.ndarray
           1D array of interpolated SNR values with the same length as the input batches.








   .. rubric:: Notes

   - All batch arrays must have the same length.
   - Uses JAX's vmap for efficient vectorized computation.
   - Calls cubic_spline_4d_jax internally for each set of parameters.





   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_aligned_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr)

   
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



.. py:function:: get_interpolated_snr_no_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr)

   
   Function to calculate the in terpolated snr for a given set of parameters
















   ..
       !! processed by numpydoc !!

