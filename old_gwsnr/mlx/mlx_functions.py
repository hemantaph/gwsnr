"""
MLX-compiled functions for gravitational wave data analysis.

This module provides high-performance MLX implementations of core functions used in
gravitational wave signal-to-noise ratio (SNR) calculations and parameter estimation.
Key features include:

- Chirp time calculations using 3.5 post-Newtonian approximations
- Antenna response pattern computations for gravitational wave detectors
- Polarization tensor calculations for plus and cross modes
- Coordinate transformations between celestial and detector frames
- Vectorized operations for efficient batch processing
- Automatic parallelization through MLX's vmap for multi-dimensional arrays

All functions are compiled with MLX's @mx.compile decorator for optimal performance and efficient computation on Apple silicon's unified memory. The implementations are optimized for use in Bayesian inference pipelines and matched filtering applications in gravitational wave astronomy.
"""

import mlx.core as mx

@mx.compile
def findchirp_chirptime_mlx(m1, m2, fmin):
    """
    Function to calculate the chirp time from minimum frequency to last stable orbit (MLX implementation).
    
    Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.

    Parameters
    ----------
    m1 : `float` or `mx.array`
        Mass of the first body in solar masses.
    m2 : `float` or `mx.array`
        Mass of the second body in solar masses.
    fmin : `float` or `mx.array`
        Lower frequency cutoff in Hz.

    Returns
    -------
    chirp_time : `mx.array`
        Time taken from f_min to f_lso (last stable orbit frequency) in seconds.

    Notes
    -----
    Calculates chirp time using 3.5PN approximation for gravitational wave Fourier phase.
    The time represents frequency evolution from fmin to last stable orbit frequency.
    Uses post-Newtonian expansion coefficients optimized for efficient MLX computation.
    MLX implementation supports JIT compilation.
    """

    Gamma = 0.5772156649015329
    # Use mx.pi, which will use the default dtype (float32)
    Pi = mx.pi
    MTSUN_SI = 4.925491025543576e-06
    
    # Ensure inputs are MLX arrays for computation, using the default dtype
    m1 = mx.array(m1)
    m2 = mx.array(m2)
    fmin = mx.array(fmin)

    # variables used to compute chirp time
    m = m1 + m2
    eta = m1 * m2 / m / m
    
    # PN Coefficients
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
        + mx.log(4.) * 6848.0 / 105.0
    )
    c6LogT = 6848.0 / 105.0

    c5T = 13.0 * Pi * eta / 3.0 - 7729.0 * Pi / 252.0
    c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0)
    c3T = -32.0 * Pi / 5.0
    c2T = 743.0 / 252.0 + eta * 11.0 / 3.0
    c0T = 5.0 * m * MTSUN_SI / (256.0 * eta)

    # This is the PN parameter v evaluated at the lower freq. cutoff
    # Use mx.power instead of jnp.power
    xT = mx.power(Pi * m * MTSUN_SI * fmin, 1.0 / 3.0)
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
    chirp_time = (
        c0T
        * (
            1
            + c2T * x2T
            + c3T * x3T
            + c4T * x4T
            + c5T * x5T
            # Use mx.log instead of jnp.log
            + (c6T + c6LogT * mx.log(xT)) * x6T
            + c7T * x7T
        )
        / x8T
    )
    return chirp_time

@mx.compile
def trunc_mlx(x):
    return mx.where(x >= 0, mx.floor(x), mx.ceil(x))

@mx.compile
def fmod_mlx(x, y):
    return x - y * trunc_mlx(x / y)

@mx.compile
def gps_to_gmst(gps_time):
    """
    Function to convert GPS time to Greenwich Mean Sidereal Time (GMST) (JAX implementation).

    Parameters
    ----------
    gps_time : `float`
        GPS time in seconds.

    Returns
    -------
    gmst : `float`
        Greenwich Mean Sidereal Time in radians.

    Notes
    -----
    Uses a linear approximation with a reference time and slope to compute GMST.
    The reference time (time0) is 1126259642.413 seconds and the slope is
    7.292115855382993e-05 radians per second, which approximates Earth's rotation rate.
    JAX implementation supports automatic differentiation for gradient-based optimization.
    """
    slope = 7.292115855382993e-05
    time0 = 1126259642.413
    time = gps_time - time0
    return slope * time + 36137.068361399164

@mx.compile
def ra_dec_to_theta_phi(ra, dec, gmst):
    """
    Function to convert right ascension and declination to spherical coordinates (JAX implementation).

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    gmst : `float`
        Greenwich Mean Sidereal Time in radians.

    Returns
    -------
    theta : `float`
        Polar angle (colatitude) in radians, measured from the north pole.
    phi : `float`
        Azimuthal angle in radians, adjusted for Earth's rotation.

    Notes
    -----
    Converts celestial coordinates (ra, dec) to spherical coordinates (theta, phi)
    in the detector frame. The azimuthal angle is corrected for Earth's rotation
    using GMST. Theta represents the angle from the north pole (colatitude).
    JAX implementation provides automatic differentiation capabilities for
    parameter estimation and optimization workflows.
    """
    phi = ra - gmst
    theta = mx.pi / 2.0 - dec
    return theta, phi

@mx.compile
def get_polarization_tensor_plus(ra, dec, time, psi):
    """
    Function to calculate the plus polarization tensor for gravitational wave detection (JAX implementation).

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    time : `float`
        GPS time of the source in seconds.
    psi : `float`
        Polarization angle of the source in radians.

    Returns
    -------
    polarization_tensor_plus : `jax.numpy.ndarray`
        3x3 plus polarization tensor matrix (m⊗m - n⊗n).

    Notes
    -----
    Calculates the plus polarization tensor in the detector frame by first converting
    celestial coordinates to spherical coordinates using GMST, then computing
    the basis vectors m and n based on the polarization angle psi. Returns the
    tensor m⊗m - n⊗n for plus polarization mode. JAX implementation supports
    automatic differentiation and GPU acceleration for efficient computation.
    """
    gmst = fmod_mlx(gps_to_gmst(time), 2 * mx.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = mx.array([mx.cos(phi) * mx.cos(theta), mx.cos(theta) * mx.sin(phi), -mx.sin(theta)])
    v = mx.array([-mx.sin(phi), mx.cos(phi), 0.])
    m = -u * mx.sin(psi) - v * mx.cos(psi)
    n = -u * mx.cos(psi) + v * mx.sin(psi)

    return mx.outer(m, m) - mx.outer(n, n)
    
@mx.compile
def get_polarization_tensor_cross(ra, dec, time, psi):
    """
    Function to calculate the cross polarization tensor for gravitational wave detection (JAX implementation).

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    time : `float`
        GPS time of the source in seconds.
    psi : `float`
        Polarization angle of the source in radians.

    Returns
    -------
    polarization_tensor_cross : `jax.numpy.ndarray`
        3x3 cross polarization tensor matrix (m⊗n + n⊗m).

    Notes
    -----
    Calculates the cross polarization tensor in the detector frame by first converting
    celestial coordinates to spherical coordinates using GMST, then computing
    the basis vectors m and n based on the polarization angle psi. Returns the
    tensor m⊗n + n⊗m for cross polarization mode. JAX implementation supports
    automatic differentiation and GPU acceleration for efficient computation.
    """
    gmst = mx.fmod(gps_to_gmst(time), 2 * mx.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = mx.array([mx.cos(phi) * mx.cos(theta), mx.cos(theta) * mx.sin(phi), -mx.sin(theta)])
    v = mx.array([-mx.sin(phi), mx.cos(phi), 0.])
    m = -u * mx.sin(psi) - v * mx.cos(psi)
    n = -u * mx.cos(psi) + v * mx.sin(psi)

    return mx.outer(m, n) + mx.outer(n, m)

@mx.compile
def antenna_response_plus(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the plus polarization antenna response for gravitational wave detection (JAX implementation).

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    time : `float`
        GPS time of the source in seconds.
    psi : `float`
        Polarization angle of the source in radians.
    detector_tensor : `jax.numpy.ndarray`
        Detector tensor for the detector (3x3 matrix).

    Returns
    -------
    antenna_response_plus : `float`
        Plus polarization antenna response of the detector.

    Notes
    -----
    Computes the plus polarization antenna response by calculating the Frobenius
    inner product between the detector tensor and the plus polarization tensor.
    The polarization tensor is determined by the source location (ra, dec),
    observation time, and polarization angle (psi). JAX implementation provides
    automatic differentiation for parameter estimation workflows.
    """
    polarization_tensor = get_polarization_tensor_plus(ra, dec, time, psi)
    return mx.sum(detector_tensor * polarization_tensor)

@mx.compile
def antenna_response_cross(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the cross polarization antenna response for gravitational wave detection (JAX implementation).

    Parameters
    ----------
    ra : `float`
        Right ascension of the source in radians.
    dec : `float`
        Declination of the source in radians.
    time : `float`
        GPS time of the source in seconds.
    psi : `float`
        Polarization angle of the source in radians.
    detector_tensor : `jax.numpy.ndarray`
        Detector tensor for the detector (3x3 matrix).

    Returns
    -------
    antenna_response_cross : `float`
        Cross polarization antenna response of the detector.

    Notes
    -----
    Computes the cross polarization antenna response by calculating the Frobenius
    inner product between the detector tensor and the cross polarization tensor.
    The polarization tensor is determined by the source location (ra, dec),
    observation time, and polarization angle (psi). JAX implementation provides
    automatic differentiation for parameter estimation workflows.
    """
    polarization_tensor = get_polarization_tensor_cross(ra, dec, time, psi)
    return mx.sum(detector_tensor * polarization_tensor)

@mx.compile
def antenna_response_array(ra, dec, time, psi, detector_tensor):
    """
    Function to calculate the antenna response for multiple detectors and sources (JAX implementation).

    Parameters
    ----------
    ra : `jax.numpy.ndarray`
        Array of right ascension values for sources in radians.
    dec : `jax.numpy.ndarray`
        Array of declination values for sources in radians.
    time : `jax.numpy.ndarray`
        Array of GPS times for sources in seconds.
    psi : `jax.numpy.ndarray`
        Array of polarization angles for sources in radians.
    detector_tensor : `jax.numpy.ndarray`
        Detector tensor array for multiple detectors (n×3×3 matrix), where n is the number of detectors.

    Returns
    -------
    Fp : `jax.numpy.ndarray`
        Plus polarization antenna response array with shape (n_detectors, n_sources).
    Fc : `jax.numpy.ndarray`
        Cross polarization antenna response array with shape (n_detectors, n_sources).

    Notes
    -----
    Computes antenna responses for both plus and cross polarizations across multiple
    detectors and source parameters simultaneously. Uses JAX's vmap for efficient
    vectorized computation with automatic differentiation support. Each antenna
    response is calculated using the Frobenius inner product between detector
    tensors and polarization tensors derived from source sky location and
    polarization angle. Optimized for GPU acceleration and gradient-based optimization.
    """

    # VMAP over detector and parameter axes
    # Outputs shape (n_det, n_param)
    Fp = mx.vmap(
        lambda d: mx.vmap(
            lambda ra_i, dec_i, time_i, psi_i: antenna_response_plus(
                ra_i, dec_i, time_i, psi_i, d
            )
        )(ra, dec, time, psi)
    )(detector_tensor)
    Fc = mx.vmap(
        lambda d: mx.vmap(
            lambda ra_i, dec_i, time_i, psi_i: antenna_response_cross(
                ra_i, dec_i, time_i, psi_i, d
            )
        )(ra, dec, time, psi)
    )(detector_tensor)
    return Fp, Fc