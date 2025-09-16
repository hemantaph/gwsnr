# -*- coding: utf-8 -*-
"""
Numba-compiled helper functions for gravitational wave signal-to-noise ratio calculations.

This module provides optimized numerical functions for gravitational wave data analysis,
including chirp time calculations, antenna response computations, polarization tensors,
coordinate transformations, and noise-weighted inner products. All functions are compiled
with Numba's @njit decorator for high-performance computation, with parallel processing
support using prange for multi-threaded execution where applicable.
"""

# -*- coding: utf-8 -*-
"""
Helper functions for gwsnr. All functions are njit compiled.
"""

import numpy as np
from numba import njit, prange

Gamma = 0.5772156649015329
Pi = np.pi
MTSUN_SI = 4.925491025543576e-06

@njit
def findchirp_chirptime(m1, m2, fmin):
    """
    Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.

    Parameters
    ----------
    m1 : `float`
        Mass of the first body in solar masses.
    m2 : `float`
        Mass of the second body in solar masses.
    fmin : `float`
        Lower frequency cutoff.

    Returns
    -------
    chirp_time : float
        Time taken from f_min to f_lso (last stable orbit frequency).
    """

    # variables used to compute chirp time
    m = m1 + m2
    eta = m1 * m2 / m / m
    c0T = c2T = c3T = c4T = c5T = c6T = c6LogT = c7T = 0.0

    c7T = Pi * (
        14809.0 * eta * eta / 378.0 - 75703.0 * eta / 756.0 - 15419335.0 / 127008.0
    )

    c6T = (
        Gamma * 6848.0 / 105.0
        - 10052469856691.0 / 23471078400.0
        + Pi * Pi * 128.0 / 3.0
        + eta * (3147553127.0 / 3048192.0 - Pi * Pi * 451.0 / 12.0)
        - eta * eta * 15211.0 / 1728.0
        + eta * eta * eta * 25565.0 / 1296.0
        + eta * eta * eta * 25565.0 / 1296.0
        + np.log(4.0) * 6848.0 / 105.0
    )
    c6LogT = 6848.0 / 105.0

    c5T = 13.0 * Pi * eta / 3.0 - 7729.0 * Pi / 252.0

    c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0)
    c3T = -32.0 * Pi / 5.0
    c2T = 743.0 / 252.0 + eta * 11.0 / 3.0
    c0T = 5.0 * m * MTSUN_SI / (256.0 * eta)

    # This is the PN parameter v evaluated at the lower freq. cutoff
    xT = np.power(Pi * m * MTSUN_SI * fmin, 1.0 / 3.0)
    x2T = xT * xT
    x3T = xT * x2T
    x4T = x2T * x2T
    x5T = x2T * x3T
    x6T = x3T * x3T
    x7T = x3T * x4T
    x8T = x4T * x4T

    # Computes the chirp time as tC = t(v_low)
    # tC = t(v_low) - t(v_upper) would be more
    # correct, but the difference is negligble.
    return (
        c0T
        * (
            1
            + c2T * x2T
            + c3T * x3T
            + c4T * x4T
            + c5T * x5T
            + (c6T + c6LogT * np.log(xT)) * x6T
            + c7T * x7T
        )
        / x8T
    )

@njit
def einsum1(m,n):
    """
    Function to calculate einsum of two 3x1 vectors

    Parameters
    ----------
    m : `numpy.ndarray`
        3x1 vector.
    n : `numpy.ndarray`
        3x1 vector.
        
    Returns
    -------
    ans : `numpy.ndarray`
        3x3 matrix.
    """
    ans = np.zeros((3,3))
    ans[0,0] = m[0]*n[0]
    ans[0,1] = m[0]*n[1]
    ans[0,2] = m[0]*n[2]
    ans[1,0] = m[1]*n[0]
    ans[1,1] = m[1]*n[1]
    ans[1,2] = m[1]*n[2]
    ans[2,0] = m[2]*n[0]
    ans[2,1] = m[2]*n[1]
    ans[2,2] = m[2]*n[2]
    return ans

@njit
def einsum2(m,n):
    """
    Function to calculate einsum of two 3x3 matrices

    Parameters
    ----------
    m : `numpy.ndarray`
        3x3 matrix.
    n : `numpy.ndarray`
        3x3 matrix.

    Returns
    -------
    ans : `numpy.ndarray`
        3x3 matrix.
    """
    ans = m[0,0]*n[0,0] + m[0,1]*n[0,1] + m[0,2]*n[0,2] + m[1,0]*n[1,0] + m[1,1]*n[1,1] + m[1,2]*n[1,2] + m[2,0]*n[2,0] + m[2,1]*n[2,1] + m[2,2]*n[2,2]
    return ans

@njit
def gps_to_gmst(gps_time):
    """
    Function to convert gps time to greenwich mean sidereal time

    Parameters
    ----------
    gps_time : `float`
        GPS time in seconds.

    Returns
    -------
    gmst : `float`
        Greenwich mean sidereal time in radians.
    """
    slope = 7.292115855425873e-05
    intercept = -45991.08966925838
    return slope*gps_time+intercept

@njit
def ra_dec_to_theta_phi(ra, dec, gmst):
    """
    Function to convert ra and dec to theta and phi

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    gmst : `float`
        Greenwich mean sidereal time in radians.

    Returns
    -------
    theta : `float`
        Polar angle in radians.
    phi : `float`
        Azimuthal angle in radians.
    """

    phi = ra - gmst
    theta = np.pi / 2.0 - dec
    return theta, phi

@njit
def get_polarization_tensor_plus(ra, dec, time, psi):
    """
    Function to calculate the polarization tensor

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    time : `float`
        GPS time of the source.
    psi : `float`
        Polarization angle of the source.

    Returns
    -------
    polarization_tensor: `numpy.ndarray`
        Polarization tensor of the detector.
    """
    gmst = np.fmod(gps_to_gmst(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    return einsum1(m, m) - einsum1(n, n)
    
@njit
def get_polarization_tensor_cross(ra, dec, time, psi):
    """
    Function to calculate the polarization tensor

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    time : `float`
        GPS time of the source.
    psi : `float`
        Polarization angle of the source.

    Returns
    -------
    polarization_tensor: `numpy.ndarray`
        Polarization tensor of the detector.
    """
    gmst = np.fmod(gps_to_gmst(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    return einsum1(m, n) + einsum1(n, m)

@njit
def antenna_response_plus(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the antenna response

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    time : `float`
        GPS time of the source.
    psi : `float`
        Polarization angle of the source.
    detector_tensor : array-like
        Detector tensor for the detector (3x3 matrix)
    mode : `str`
        Mode of the polarization. Default is 'plus'.

    Returns
    -------
    antenna_response: `float`
        Antenna response of the detector.
    """

    polarization_tensor = get_polarization_tensor_plus(ra, dec, time, psi)
    return einsum2(detector_tensor, polarization_tensor)

@njit
def antenna_response_cross(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the antenna response

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    time : `float`
        GPS time of the source.
    psi : `float`
        Polarization angle of the source.
    detector_tensor : array-like
        Detector tensor for the detector (3x3 matrix)
    mode : `str`
        Mode of the polarization. Default is 'plus'.

    Returns
    -------
    antenna_response: `float`
        Antenna response of the detector.
    """

    polarization_tensor = get_polarization_tensor_cross(ra, dec, time, psi)
    return einsum2(detector_tensor, polarization_tensor)

@njit(parallel=True)
def antenna_response_array(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the antenna response in array form.

    Parameters
    ----------
    ra : `numpy.ndarray`
        Right ascension of the source in radians.
    dec : `numpy.ndarray`
        Declination of the source in radians.
    time : `numpy.ndarray`
        GPS time of the source.
    psi : `numpy.ndarray`
        Polarization angle of the source.
    detector_tensor : array-like
        Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

    Returns
    -------
    antenna_response: `numpy.ndarray`
        Antenna response of the detector. Shape is (n, len(ra)).
    """

    len_det = len(detector_tensor)
    len_param = len(ra) 
    Fp = np.zeros((len_det, len_param))
    Fc = np.zeros((len_det, len_param))

    for i in prange(len_param):
        for j in range(len_det):
        
            Fp[j,i] = antenna_response_plus(ra[i], dec[i], time[i], psi[i], detector_tensor[j])
            Fc[j,i] = antenna_response_cross(ra[i], dec[i], time[i], psi[i], detector_tensor[j])

    return Fp, Fc

@njit
def effective_distance(
    luminosity_distance, theta_jn, ra, dec, geocent_time, psi, detector_tensor
):
    """
    Function to calculate the effective distance of the source.

    Parameters
    ----------
    luminosity_distance : `float`
        Luminosity distance of the source in Mpc.
    theta_jn : `float`
        Angle between the line of sight and the orbital angular momentum vector.
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    time : `float`
        GPS time of the source.
    psi : `float`
        Polarization angle of the source.
    detector_tensor : array-like
        Detector tensor for the detector (3x3 matrix).

    Returns
    -------
    effective_distance: `float`
        Effective distance of the source in Mpc.
    """

    Fp, Fc = antenna_response_plus(ra, dec, geocent_time, psi, detector_tensor), antenna_response_cross(ra, dec, geocent_time, psi, detector_tensor)

    return luminosity_distance / np.sqrt(
            Fp**2 * ((1 + np.cos(theta_jn) ** 2) / 2) ** 2
            + Fc**2 * np.cos(theta_jn) ** 2
        )

@njit(parallel=True)
def effective_distance_array(
    luminosity_distance, theta_jn, ra, dec, geocent_time, psi, detector_tensor
):
    """
    Function to calculate the effective distance of the source in array form.

    Parameters
    ----------
    luminosity_distance : `numpy.ndarray`
        Luminosity distance of the source in Mpc.
    theta_jn : `numpy.ndarray`
        Angle between the line of sight and the orbital angular momentum vector.
    ra : `numpy.ndarray`
        Right ascension of the source in radians.
    dec : `numpy.ndarray`
        Declination of the source in radians.
    time : `numpy.ndarray`
        GPS time of the source.
    psi : `numpy.ndarray`
        Polarization angle of the source.
    detector_tensor : array-like
        Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

    Returns
    -------
    effective_distance: `numpy.ndarray`
        Effective distance of the source in Mpc. Shape is (n, len(ra)).
    """

    len_det = len(detector_tensor)
    len_param = len(ra) 
    eff_dist = np.zeros((len_det, len_param))

    for i in prange(len_param):
        for j in range(len_det):
            eff_dist[j,i] = effective_distance(
                luminosity_distance[i], theta_jn[i], ra[i], dec[i], geocent_time[i], psi[i], detector_tensor[j]
            )

    return eff_dist

@njit
def noise_weighted_inner_product(
    signal1, signal2, psd, duration,
):
    """
    Noise weighted inner product of two time series data sets.

    Parameters
    ----------
    signal1: `numpy.ndarray` or `float`
        First series data set.
    signal2: `numpy.ndarray` or `float`
        Second series data set.
    psd: `numpy.ndarray` or `float`
        Power spectral density of the detector.
    duration: `float`
        Duration of the data.
    """

    nwip_arr = np.conj(signal1) * signal2 / psd
    return 4 / duration * np.sum(nwip_arr)

@njit(parallel=True)
def linear_interpolator(xnew_array, y_array, x_array, fill_value=np.inf):
    """
    Linear interpolator for 1D data.

    Parameters
    ----------
    xnew_array : `numpy.ndarray`
        New x values to interpolate.
    y_array : `numpy.ndarray`
        y values corresponding to the x_array.
    x_array : `numpy.ndarray`
        Original x values.

    Returns
    -------
    result : `numpy.ndarray`
        Interpolated y values at xnew_array.
    """

    result = np.zeros_like(xnew_array)
    len_ = xnew_array.shape[0]

    for j in prange(len_):
        xnew = xnew_array[j]
        # Handling extrapolation
        i = np.searchsorted(x_array, xnew) - 1
        # Linear interpolation
        if (i < len(x_array) - 1) and (i > 0):
            x0, x1 = x_array[i], x_array[i + 1]
            y0, y1 = y_array[i], y_array[i + 1]
            result[j] = y0 + (y1 - y0) * (xnew - x0) / (x1 - x0)
        else:
            result[j] = fill_value

    return result

# @njit
# def _helper_hphc(hp,hc,fsize_arr,fs,size,f_l,i):
#     # remove the np.nan padding
#     hp_ = np.array(hp[i][:fsize_arr[i]], dtype=np.complex128)
#     hc_ = np.array(hc[i][:fsize_arr[i]], dtype=np.complex128)
#     # find the index of 20Hz or nearby
#     # set all elements to zero below this index
#     idx = np.abs(fs[i] - f_l).argmin()
#     hp_[i][0:idx] = 0.0 + 0.0j
#     hc_[i][0:idx] = 0.0 + 0.0j

#     return hp_,hc_
