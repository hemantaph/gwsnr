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
   gwsnr.numba.antenna_response_plus
   gwsnr.numba.antenna_response_cross
   gwsnr.numba.antenna_response_array
   gwsnr.numba.noise_weighted_inner_product
   gwsnr.numba.effective_distance
   gwsnr.numba.effective_distance_array
   gwsnr.numba.cubic_spline_interpolator
   gwsnr.numba.get_interpolated_snr_aligned_spins_numba
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

.. py:function:: cubic_spline_interpolator(xnew_array, coefficients, x)

   
   Function to calculate the interpolated value of snr_halfscaled given the total mass (xnew). This is based off 1D cubic spline interpolation.


   :Parameters:

       **xnew_array** : `numpy.ndarray`
           Total mass of the binary.

       **coefficients** : `numpy.ndarray`
           Array of coefficients for the cubic spline interpolation.

       **x** : `numpy.ndarray`
           Array of total mass values for the coefficients.

   :Returns:

       **result** : `float`
           Interpolated value of snr_halfscaled.













   ..
       !! processed by numpydoc !!



