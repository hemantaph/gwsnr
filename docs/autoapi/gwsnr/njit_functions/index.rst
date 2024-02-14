:py:mod:`gwsnr.njit_functions`
==============================

.. py:module:: gwsnr.njit_functions

.. autoapi-nested-parse::

   Helper functions for gwsnr. All functions are njit compiled.

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.njit_functions.findchirp_chirptime
   gwsnr.njit_functions.einsum1
   gwsnr.njit_functions.einsum2
   gwsnr.njit_functions.gps_to_gmst
   gwsnr.njit_functions.ra_dec_to_theta_phi
   gwsnr.njit_functions.get_polarization_tensor
   gwsnr.njit_functions.antenna_response
   gwsnr.njit_functions.antenna_response_array
   gwsnr.njit_functions.noise_weighted_inner_product
   gwsnr.njit_functions.get_interpolated_snr
   gwsnr.njit_functions.cubic_spline_interpolator2d
   gwsnr.njit_functions.cubic_spline_interpolator
   gwsnr.njit_functions.coefficients_generator



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

.. py:function:: get_polarization_tensor(ra, dec, time, psi, mode='plus')

   
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

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       polarization_tensor: `numpy.ndarray`
           Polarization tensor of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response(ra, dec, time, psi, detector_tensor, mode='plus')

   
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

.. py:function:: get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_halfscaled, ratio_arr, mtot_arr)

   
   Function to calculate the interpolated snr for a given set of parameters


   :Parameters:

       **mass_1** : `numpy.ndarray`
           Mass of the first body in solar masses.

       **mass_2** : `numpy.ndarray`
           Mass of the second body in solar masses.

       **luminosity_distance** : `float`
           Luminosity distance to the source in Mpc.

       **theta_jn** : `numpy.ndarray`
           Angle between the total angular momentum and the line of sight to the source in radians.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **geocent_time** : `numpy.ndarray`
           GPS time of the source.

       **ra** : ``numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **snr_halfscaled** : `numpy.ndarray`
           Array of snr_halfscaled coefficients for the detector.

       **ratio_arr** : `numpy.ndarray`
           Array of mass ratio values for the snr_halfscaled coefficients.

       **mtot_arr** : `numpy.ndarray`
           Array of total mass values for the snr_halfscaled coefficients.

   :Returns:

       **snr** : `float`
           snr of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_interpolator2d(xnew, ynew, coefficients, x, y)

   
   Function to calculate the interpolated value of snr_halfscaled given the mass ratio (ynew) and total mass (xnew). This is based off 2D bicubic spline interpolation.


   :Parameters:

       **xnew** : `float`
           Total mass of the binary.

       **ynew** : `float`
           Mass ratio of the binary.

       **coefficients** : `numpy.ndarray`
           Array of coefficients for the cubic spline interpolation.

       **x** : `numpy.ndarray`
           Array of total mass values for the coefficients.

       **y** : `numpy.ndarray`
           Array of mass ratio values for the coefficients.

   :Returns:

       **result** : `float`
           Interpolated value of snr_halfscaled.













   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_interpolator(xnew, coefficients, x)

   
   Function to calculate the interpolated value of snr_halfscaled given the total mass (xnew). This is based off 1D cubic spline interpolation.


   :Parameters:

       **xnew** : `float`
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

.. py:function:: coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4)

   
   Function to generate the coefficients for the cubic spline interpolation of fn(y)=z.


   :Parameters:

       **y1, y2, y3, y4, z1, z2, z3, z4: `float`**
           Values of y and z for the cubic spline interpolation.

   :Returns:

       coefficients: `numpy.ndarray`
           Coefficients for the cubic spline interpolation.













   ..
       !! processed by numpydoc !!

