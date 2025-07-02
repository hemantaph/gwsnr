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


.. py:function:: cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i)

   
   Catmull-Rom spline interpolation (Numba-friendly, 4 points).
   x_pts: 4 points, assumed sorted, uniform or non-uniform spacing.
   y_pts: values at those points.
   x_eval: point at which to evaluate.
   condition_i: for edge handling.
















   ..
       !! processed by numpydoc !!







