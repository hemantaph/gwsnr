:py:mod:`gwsnr.numba.njit_functions`
====================================

.. py:module:: gwsnr.numba.njit_functions

.. autoapi-nested-parse::

   Numba-compiled helper functions for gravitational wave signal-to-noise ratio calculations.

   This module provides optimized numerical functions for gravitational wave data analysis,
   including chirp time calculations, antenna response computations, polarization tensors,
   coordinate transformations, and noise-weighted inner products. All functions are compiled
   with Numba's @njit decorator for high-performance computation, with parallel processing
   support using prange for multi-threaded execution where applicable.

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.numba.njit_functions.findchirp_chirptime
   gwsnr.numba.njit_functions.einsum1
   gwsnr.numba.njit_functions.einsum2
   gwsnr.numba.njit_functions.gps_to_gmst
   gwsnr.numba.njit_functions.ra_dec_to_theta_phi
   gwsnr.numba.njit_functions.get_polarization_tensor_plus
   gwsnr.numba.njit_functions.get_polarization_tensor_cross
   gwsnr.numba.njit_functions.antenna_response_plus
   gwsnr.numba.njit_functions.antenna_response_cross
   gwsnr.numba.njit_functions.antenna_response_array
   gwsnr.numba.njit_functions.effective_distance
   gwsnr.numba.njit_functions.effective_distance_array
   gwsnr.numba.njit_functions.noise_weighted_inner_product
   gwsnr.numba.njit_functions.linear_interpolator



Attributes
~~~~~~~~~~

.. autoapisummary::

   gwsnr.numba.njit_functions.Gamma
   gwsnr.numba.njit_functions.Pi
   gwsnr.numba.njit_functions.MTSUN_SI


.. py:data:: Gamma
   :value: '0.5772156649015329'

   

.. py:data:: Pi

   

.. py:data:: MTSUN_SI
   :value: '4.925491025543576e-06'

   

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

.. py:function:: linear_interpolator(xnew_array, y_array, x_array, fill_value=np.inf)

   
   Linear interpolator for 1D data.


   :Parameters:

       **xnew_array** : `numpy.ndarray`
           New x values to interpolate.

       **y_array** : `numpy.ndarray`
           y values corresponding to the x_array.

       **x_array** : `numpy.ndarray`
           Original x values.

   :Returns:

       **result** : `numpy.ndarray`
           Interpolated y values at xnew_array.













   ..
       !! processed by numpydoc !!

