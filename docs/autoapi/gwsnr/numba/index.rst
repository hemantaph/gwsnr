:orphan:

:py:mod:`gwsnr.numba`
=====================

.. py:module:: gwsnr.numba


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   njit_functions/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.numba.findchirp_chirptime
   gwsnr.numba.einsum1
   gwsnr.numba.einsum2
   gwsnr.numba.gps_to_gmst
   gwsnr.numba.ra_dec_to_theta_phi
   gwsnr.numba.get_polarization_tensor_plus
   gwsnr.numba.get_polarization_tensor_cross
   gwsnr.numba.antenna_response_plus
   gwsnr.numba.antenna_response_cross
   gwsnr.numba.antenna_response_array
   gwsnr.numba.noise_weighted_inner_product
   gwsnr.numba.effective_distance
   gwsnr.numba.effective_distance_array
   gwsnr.numba.find_index_1d_numba
   gwsnr.numba.cubic_function_4pts_numba
   gwsnr.numba.cubic_spline_4d_numba
   gwsnr.numba.cubic_spline_4d_batched_numba
   gwsnr.numba.get_interpolated_snr_aligned_spins_numba
   gwsnr.numba.cubic_spline_2d_numba
   gwsnr.numba.cubic_spline_2d_batched_numba
   gwsnr.numba.get_interpolated_snr_no_spins_numba



.. py:function:: findchirp_chirptime(m1, m2, fmin)

   
   Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.


   :Parameters:

       **m1** : `float`
           Mass of the first body in solar masses.

       **m2** : `float`
           Mass of the second body in solar masses.

       **fmin** : `float`
           Lower frequency cutoff.

   :Returns:

       **chirp_time** : float
           Time taken from f_min to f_lso (last stable orbit frequency).













   ..
       !! processed by numpydoc !!

.. py:function:: einsum1(m, n)

   
   Function to calculate einsum of two 3x1 vectors


   :Parameters:

       **m** : `numpy.ndarray`
           3x1 vector.

       **n** : `numpy.ndarray`
           3x1 vector.

   :Returns:

       **ans** : `numpy.ndarray`
           3x3 matrix.













   ..
       !! processed by numpydoc !!

.. py:function:: einsum2(m, n)

   
   Function to calculate einsum of two 3x3 matrices


   :Parameters:

       **m** : `numpy.ndarray`
           3x3 matrix.

       **n** : `numpy.ndarray`
           3x3 matrix.

   :Returns:

       **ans** : `numpy.ndarray`
           3x3 matrix.













   ..
       !! processed by numpydoc !!

.. py:function:: gps_to_gmst(gps_time)

   
   Function to convert gps time to greenwich mean sidereal time


   :Parameters:

       **gps_time** : `float`
           GPS time in seconds.

   :Returns:

       **gmst** : `float`
           Greenwich mean sidereal time in radians.













   ..
       !! processed by numpydoc !!

.. py:function:: ra_dec_to_theta_phi(ra, dec, gmst)

   
   Function to convert ra and dec to theta and phi


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **gmst** : `float`
           Greenwich mean sidereal time in radians.

   :Returns:

       **theta** : `float`
           Polar angle in radians.

       **phi** : `float`
           Azimuthal angle in radians.













   ..
       !! processed by numpydoc !!

.. py:function:: get_polarization_tensor_plus(ra, dec, time, psi)

   
   Function to calculate the polarization tensor


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

   :Returns:

       polarization_tensor: `numpy.ndarray`
           Polarization tensor of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: get_polarization_tensor_cross(ra, dec, time, psi)

   
   Function to calculate the polarization tensor


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

   :Returns:

       polarization_tensor: `numpy.ndarray`
           Polarization tensor of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_plus(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       antenna_response: `float`
           Antenna response of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_cross(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       antenna_response: `float`
           Antenna response of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_array(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response in array form.


   :Parameters:

       **ra** : `numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **time** : `numpy.ndarray`
           GPS time of the source.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

   :Returns:

       antenna_response: `numpy.ndarray`
           Antenna response of the detector. Shape is (n, len(ra)).













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_product(signal1, signal2, psd, duration)

   
   Noise weighted inner product of two time series data sets.


   :Parameters:

       **signal1: `numpy.ndarray` or `float`**
           First series data set.

       **signal2: `numpy.ndarray` or `float`**
           Second series data set.

       **psd: `numpy.ndarray` or `float`**
           Power spectral density of the detector.

       **duration: `float`**
           Duration of the data.














   ..
       !! processed by numpydoc !!

.. py:function:: effective_distance(luminosity_distance, theta_jn, ra, dec, geocent_time, psi, detector_tensor)

   
   Function to calculate the effective distance of the source.


   :Parameters:

       **luminosity_distance** : `float`
           Luminosity distance of the source in Mpc.

       **theta_jn** : `float`
           Angle between the line of sight and the orbital angular momentum vector.

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix).

   :Returns:

       effective_distance: `float`
           Effective distance of the source in Mpc.













   ..
       !! processed by numpydoc !!

.. py:function:: effective_distance_array(luminosity_distance, theta_jn, ra, dec, geocent_time, psi, detector_tensor)

   
   Function to calculate the effective distance of the source in array form.


   :Parameters:

       **luminosity_distance** : `numpy.ndarray`
           Luminosity distance of the source in Mpc.

       **theta_jn** : `numpy.ndarray`
           Angle between the line of sight and the orbital angular momentum vector.

       **ra** : `numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **time** : `numpy.ndarray`
           GPS time of the source.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

   :Returns:

       effective_distance: `numpy.ndarray`
           Effective distance of the source in Mpc. Shape is (n, len(ra)).













   ..
       !! processed by numpydoc !!

.. py:function:: find_index_1d_numba(x_array, x_new)

   
   Find the index for cubic spline interpolation in 1D.
   Returns the index and a condition for edge handling.


   :Parameters:

       **x_array** : jnp.ndarray
           The array of x values for interpolation. Must be sorted in ascending order.

       **x_new** : float or jnp.ndarray
           The new x value(s) to find the index for.

   :Returns:

       **i** : int
           The index in `x_array` such that `x_array[i-1] <= x_new < x_array[i+1]`.

       **condition_i** : int
           An integer indicating the condition for edge handling:
           - 1: `x_new` is less than or equal to the first element of `x_array`.
           - 2: `x_new` is between the first and last elements of `x_array`.
           - 3: `x_new` is greater than or equal to the last element of `x_array`.








   .. rubric:: Notes

   Uses binary search with clipped indices to ensure valid 4-point stencils.
   The condition parameter determines linear vs cubic interpolation at boundaries.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i)

   
   Evaluate a cubic function at a given point using 4-point interpolation.


   :Parameters:

       **x_eval** : float
           The x value at which to evaluate the cubic function.

       **x_pts** : jnp.ndarray
           The x values of the 4 points used for interpolation. Must be sorted in ascending order

       **y_pts** : jnp.ndarray
           The y values corresponding to the x_pts. Must have the same length as x_pts.

       **condition_i** : int
           An integer indicating the condition for edge handling:
           - 1: `x_eval` is less than or equal to the first element of `x_pts`.
           - 2: `x_eval` is between the first and last elements of `x_pts`.
           - 3: `x_eval` is greater than or equal to the last element of `x_pts`.

   :Returns:

       float
           The interpolated value at `x_eval`.








   .. rubric:: Notes

   This function uses cubic Hermite interpolation for the main case (condition_i == 2).
   For edge cases (condition_i == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.
   The x_pts and y_pts must be of length 4, and x_pts must be sorted in ascending order.
   The function assumes that the input arrays are valid and does not perform additional checks.
   If the input arrays are not of length 4, or if x_pts is not sorted, the behavior is undefined.
   The function is designed to be used with Numba's JIT compilation for performance.
   It is optimized for speed and does not include error handling or input validation.
   The cubic Hermite interpolation is based on the tangents calculated from the y values at the four points.
   The tangents are computed using the differences between the y values and the x values
   to ensure smoothness and continuity of the interpolated curve.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new, int_q, int_m, int_a1, int_a2)

   
   Perform cubic spline interpolation in 4D for the given arrays and new values.


   :Parameters:

       **q_array** : jnp.ndarray
           The array of q values for interpolation. Must be sorted in ascending order.

       **mtot_array** : jnp.ndarray
           The array of mtot values for interpolation. Must be sorted in ascending order.

       **a1_array** : jnp.ndarray
           The array of a1 values for interpolation. Must be sorted in ascending order.

       **a2_array** : jnp.ndarray
           The array of a2 values for interpolation. Must be sorted in ascending order.

       **snrpartialscaled_array** : jnp.ndarray
           The 4D array of snrpartialscaled values with shape (4, 4, 4, 4).
           This array contains the values to be interpolated.

       **q_new** : float
           The new q value at which to evaluate the cubic spline.

       **mtot_new** : float
           The new mtot value at which to evaluate the cubic spline.

       **a1_new** : float
           The new a1 value at which to evaluate the cubic spline.

       **a2_new** : float
           The new a2 value at which to evaluate the cubic spline.

       **int_q** : int
           edge condition for q interpolation. Refer to `find_index_1d_numba` for details.

       **int_m** : int
           edge condition for mtot interpolation. Refer to `find_index_1d_numba` for details.

       **int_a1** : int
           edge condition for a1 interpolation. Refer to `find_index_1d_numba` for details.

       **int_a2** : int
           edge condition for a2 interpolation. Refer to `find_index_1d_numba` for details.

   :Returns:

       float
           The interpolated value at the new coordinates (q_new, mtot_new, a1_new, a2_new).








   .. rubric:: Notes

   This function uses cubic Hermite interpolation for the main case (int_q == 2, int_m == 2, int_a1 == 2, int_a2 == 2).
   For edge cases (int_q == 1 or 3, int_m == 1 or 3, int_a1 == 1 or 3, int_a2 == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_batched_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)

   
   Perform cubic spline interpolation in 4D for a batch of new values.


   :Parameters:

       **q_array** : jnp.ndarray
           The array of q values for interpolation. Must be sorted in ascending order.

       **mtot_array** : jnp.ndarray
           The array of mtot values for interpolation. Must be sorted in ascending order.

       **a1_array** : jnp.ndarray
           The array of a1 values for interpolation. Must be sorted in ascending order.

       **a2_array** : jnp.ndarray
           The array of a2 values for interpolation. Must be sorted in ascending order.

       **snrpartialscaled_array** : jnp.ndarray
           The 4D array of snrpartialscaled values.

       **q_new_batch** : jnp.ndarray
           The new q values at which to evaluate the cubic spline. Must be a 1D array.

       **mtot_new_batch** : jnp.ndarray
           The new mtot values at which to evaluate the cubic spline. Must be a 1
           The new a1 values at which to evaluate the cubic spline. Must be a 1D array.

       **a1_new_batch** : jnp.ndarray
           The new a1 values at which to evaluate the cubic spline. Must be a 1D array.

       **a2_new_batch** : jnp.ndarray
           The new a2 values at which to evaluate the cubic spline. Must be a 1D array.

   :Returns:

       jnp.ndarray
           A 1D array of interpolated values at the new coordinates (q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch).













   ..
       !! processed by numpydoc !!





