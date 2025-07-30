:orphan:

:py:mod:`gwsnr.jax.jaxjit_interpolators_old`
============================================

.. py:module:: gwsnr.jax.jaxjit_interpolators_old


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.jax.jaxjit_interpolators_old.find_index_1d_jax
   gwsnr.jax.jaxjit_interpolators_old.cubic_function_4pts_jax
   gwsnr.jax.jaxjit_interpolators_old.cubic_spline_4d_jax
   gwsnr.jax.jaxjit_interpolators_old.cubic_spline_4d_batched_jax
   gwsnr.jax.jaxjit_interpolators_old.get_interpolated_snr_aligned_spins_jax
   gwsnr.jax.jaxjit_interpolators_old.get_interpolated_snr_no_spins_jax



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

