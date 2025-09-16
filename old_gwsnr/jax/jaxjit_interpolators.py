# -*- coding: utf-8 -*-

import jax
# jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from ..numba import antenna_response_array

@jit
def find_index_1d_jax(x_array, x_new):
    """
    Find the index for cubic spline interpolation in 1D.
    Returns the index and a condition for edge handling.

    Parameters
    ----------
    x_array : jnp.ndarray
        The array of x values for interpolation. Must be sorted in ascending order.
    x_new : float or jnp.ndarray
        The new x value(s) to find the index for.

    Returns
    -------
    i : jnp.ndarray
        The index in `x_array` where `x_new` would fit, clipped to range [1, N-3] 
        where N is the length of x_array.
    condition_i : jnp.ndarray
        A condition indicating which interpolation branch to use:
        - 1: Use linear interpolation at the left edge (x_new <= x_array[1]).
        - 2: Use cubic interpolation in the middle.
        - 3: Use linear interpolation at the right edge (x_new >= x_array[N-2]).

    Notes
    -----
    Uses binary search with clipped indices to ensure valid 4-point stencils.
    The condition parameter determines linear vs cubic interpolation at boundaries.
    """
    N = x_array.shape[0]
    # Use jnp.searchsorted for robust, fast binary search.
    i = jnp.searchsorted(x_array, x_new, side='right') - 1
    i = jnp.clip(i, 1, N - 3)

    # Condition for edge handling (same as Numba version)
    condition_i = jnp.where(x_new <= x_array[1], 1,
                  jnp.where(x_new >= x_array[N-2], 3, 2))
    
    return i, condition_i

@jit
def spline_interp_4pts_jax(x_eval, x_pts, y_pts, condition_i):
    """
    Performs piecewise interpolation using 4 points with JAX compatibility.
    This function implements a piecewise interpolation scheme that uses:
    - Linear interpolation at the left boundary (condition_i=1)
    - Cubic interpolation in the middle region (condition_i=2)
    - Linear interpolation at the right boundary (condition_i=3)
    The cubic interpolation uses cubic Hermite spline coefficients for smooth
    interpolation between the middle two points, while the boundary regions
    use linear interpolation for stability.
    Parameters
    ----------
    x_eval : array_like
        The x-coordinate(s) where interpolation is to be evaluated.
    x_pts : array_like
        Array of 4 x-coordinates of the interpolation points, ordered as
        [x0, x1, x2, x3] where x1 and x2 are the main interpolation interval.
    y_pts : array_like
        Array of 4 y-coordinates corresponding to x_pts, ordered as
        [y0, y1, y2, y3].
    condition_i : int
        Interpolation mode selector:
        - 1: Linear interpolation using points (x0, y0) and (x1, y1)
        - 2: Cubic interpolation using all 4 points with x_eval in [x1, x2]
        - 3: Linear interpolation using points (x2, y2) and (x3, y3)

    Returns
    -------
    array_like
        Interpolated value(s) at x_eval using the specified interpolation method.

    Notes
    -----
    - The function handles degenerate cases where denominators are zero by
      returning appropriate fallback values (y0, y1, or y2 respectively).
    - Uses JAX's lax.switch for efficient conditional execution.
    - The cubic interpolation uses normalized parameter t = (x_eval - x1) / (x2 - x1).
    - Cubic coefficients follow the pattern: a*t³ + b*t² + c*t + d where:
    """

    # Linear at boundaries
    def left_branch(operands):
        x_eval, x_pts, y_pts = operands
        denom = x_pts[1] - x_pts[0]
        return y_pts[0] + (y_pts[1] - y_pts[0]) * (x_eval - x_pts[0]) / denom
    def right_branch(operands):
        x_eval, x_pts, y_pts = operands
        denom = x_pts[3] - x_pts[2]
        return y_pts[2] + (y_pts[3] - y_pts[2]) * (x_eval - x_pts[2]) / denom
    # Cubic interpolation in the middle
    def cubic_branch(operands):
        x_eval, x_pts, y_pts = operands
        x0, x1, x2, x3 = x_pts[0], x_pts[1], x_pts[2], x_pts[3]
        y0, y1, y2, y3 = y_pts[0], y_pts[1], y_pts[2], y_pts[3]
        denom = x2 - x1
        t = (x_eval - x1) / denom

        # --- Cubic Hermite spline tangents ---
        m1 = ((y2 - y1) / (x2 - x1)) * ((x1 - x0) / (x2 - x0)) + ((y1 - y0) / (x1 - x0)) * ((x2 - x1) / (x2 - x0))
        m2 = ((y3 - y2) / (x3 - x2)) * ((x2 - x1) / (x3 - x1)) + ((y2 - y1) / (x2 - x1)) * ((x3 - x2) / (x3 - x1))

        # --- Hermite cubic basis ---
        h00 = (2 * t**3 - 3 * t**2 + 1)
        h10 = (t**3 - 2 * t**2 + t)
        h01 = (-2 * t**3 + 3 * t**2)
        h11 = (t**3 - t**2)

        # Interpolated value
        return h00 * y1 + h10 * m1 * denom + h01 * y2 + h11 * m2 * denom
    
    return jax.lax.switch(
        condition_i - 1,  # 1 -> 0, 2 -> 1, 3 -> 2
        [left_branch, cubic_branch, right_branch],
        (x_eval, x_pts, y_pts)
    )

######### Interpolation 4D #########
@jit
def spline_interp_4x4x4x4pts_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
    """
    Function that performs the FULL 4D interpolation for a SINGLE point.
    This function finds indices, slices data, and then uses vmap internally
    to perform interpolation efficiently without Python loops.
    """
    # 1. Find indices and conditions for the new point
    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)
    a1_idx, int_a1 = find_index_1d_jax(a1_array, a1_new)
    a2_idx, int_a2 = find_index_1d_jax(a2_array, a2_new)

    # 2. Slice the small 4x4x4x4 cube of data and the corresponding 4-point coordinate arrays
    q_pts = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    m_pts = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
    a1_pts = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
    a2_pts = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))
    data_cube = lax.dynamic_slice(snrpartialscaled_array, (q_idx-1, m_idx-1, a1_idx-1, a2_idx-1), (4, 4, 4, 4))

    # 3. Perform 4D interpolation using vmap to eliminate loops
    # Interpolate along the last dimension (a2) for each of the 4x4x4 slices
    interp_along_a2 = vmap(
        lambda y_slice: spline_interp_4pts_jax(a2_new, a2_pts, y_slice, int_a2)
    )(data_cube.reshape(64, 4)).reshape(4, 4, 4)

    # Interpolate along the next dimension (a1)
    interp_along_a1 = vmap(
        lambda y_slice: spline_interp_4pts_jax(a1_new, a1_pts, y_slice, int_a1)
    )(interp_along_a2.reshape(16, 4)).reshape(4, 4)

    # Interpolate along the mtot dimension
    interp_along_m = vmap(
        lambda y_slice: spline_interp_4pts_jax(mtot_new, m_pts, y_slice, int_m)
    )(interp_along_a1)

    # Final interpolation along the q dimension
    final_snr = spline_interp_4pts_jax(q_new, q_pts, interp_along_m, int_q)

    return final_snr

@jit
def spline_interp_4x4x4x4pts_batched_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
    """
    Perform batched 4D cubic spline interpolation using JAX vectorization.
    """
    # Vectorize the complete single-point interpolation function.
    # This is the only vmap call needed at the top level.
    vmapped_interpolator = vmap(
        spline_interp_4x4x4x4pts_jax,
        in_axes=(None, None, None, None, None, 0, 0, 0, 0)
    )

    # Call the fully vectorized function on the batch data.
    return vmapped_interpolator(
        q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array,
        q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch
    )

def get_interpolated_snr_aligned_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000):
    """
    Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
    This function computes the SNR for gravitational wave signals with aligned spins across multiple 
    detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
    and combines results from multiple detectors to produce the effective SNR.

    Parameters
    ----------
    mass_1 : jax.numpy.ndarray
        Primary mass of the binary system in solar masses.
    mass_2 : jax.numpy.ndarray  
        Secondary mass of the binary system in solar masses.
    luminosity_distance : jax.numpy.ndarray
        Luminosity distance to the source in Mpc.
    theta_jn : jax.numpy.ndarray
        Inclination angle between the orbital angular momentum and line of sight in radians.
    psi : jax.numpy.ndarray
        Polarization angle in radians.
    geocent_time : jax.numpy.ndarray
        GPS time of coalescence at the geocenter in seconds.
    ra : jax.numpy.ndarray
        Right ascension of the source in radians.
    dec : jax.numpy.ndarray
        Declination of the source in radians.
    a_1 : jax.numpy.ndarray
        Dimensionless spin magnitude of the primary black hole.
    a_2 : jax.numpy.ndarray
        Dimensionless spin magnitude of the secondary black hole.
    detector_tensor : jax.numpy.ndarray
        Detector tensor array containing detector response information.
        Shape: (n_detectors, ...)
    snr_partialscaled : jax.numpy.ndarray
        Pre-computed scaled partial SNR values for interpolation.
        Shape: (n_detectors, ...)
    ratio_arr : jax.numpy.ndarray
        Mass ratio grid points for interpolation (q = m2/m1).
    mtot_arr : jax.numpy.ndarray
        Total mass grid points for interpolation.
    a1_arr : jax.numpy.ndarray
        Primary spin grid points for interpolation.
    a_2_arr : jax.numpy.ndarray
        Secondary spin grid points for interpolation.

    Returns
    -------
    snr : jax.numpy.ndarray
        SNR values for each detector. Shape: (n_detectors, n_samples)
    snr_effective : jax.numpy.ndarray
        Effective SNR combining all detectors. Shape: (n_samples,)
    snr_partial_ : jax.numpy.ndarray
        Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)
    d_eff : jax.numpy.ndarray
        Effective distance for each detector accounting for antenna response.
        Shape: (n_detectors, n_samples)

    Notes
    -----
    - Uses 4D cubic spline interpolation for efficient SNR calculation
    - Assumes aligned spins (no precession)
    - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
    - Chirp mass and inclination-dependent factors are computed analytically
    """

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    snr, snr_effective, snr_partial_, d_eff = get_interpolated_snr_aligned_spins_helper(
        jnp.array(mass_1), 
        jnp.array(mass_2), 
        jnp.array(luminosity_distance), 
        jnp.array(theta_jn), 
        jnp.array(a_1), 
        jnp.array(a_2), 
        jnp.array(snr_partialscaled), 
        jnp.array(ratio_arr), 
        jnp.array(mtot_arr), 
        jnp.array(a1_arr), 
        jnp.array(a_2_arr),
        jnp.array(Fp),
        jnp.array(Fc),
        jnp.array(detector_tensor)
    )

    return np.array(snr), np.array(snr_effective), np.array(snr_partial_), np.array(d_eff)

@jit
def get_interpolated_snr_aligned_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, a_1, a_2, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, Fp, Fc, detector_tensor):
    """
    Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
    This function computes the SNR for gravitational wave signals with aligned spins across multiple 
    detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
    and combines results from multiple detectors to produce the effective SNR.

    Parameters
    ----------
    mass_1 : jax.numpy.ndarray
        Primary mass of the binary system in solar masses.
    mass_2 : jax.numpy.ndarray  
        Secondary mass of the binary system in solar masses.
    luminosity_distance : jax.numpy.ndarray
        Luminosity distance to the source in Mpc.
    theta_jn : jax.numpy.ndarray
        Inclination angle between the orbital angular momentum and line of sight in radians.
    psi : jax.numpy.ndarray
        Polarization angle in radians.
    geocent_time : jax.numpy.ndarray
        GPS time of coalescence at the geocenter in seconds.
    ra : jax.numpy.ndarray
        Right ascension of the source in radians.
    dec : jax.numpy.ndarray
        Declination of the source in radians.
    a_1 : jax.numpy.ndarray
        Dimensionless spin magnitude of the primary black hole.
    a_2 : jax.numpy.ndarray
        Dimensionless spin magnitude of the secondary black hole.
    detector_tensor : jax.numpy.ndarray
        Detector tensor array containing detector response information.
        Shape: (n_detectors, ...)
    snr_partialscaled : jax.numpy.ndarray
        Pre-computed scaled partial SNR values for interpolation.
        Shape: (n_detectors, ...)
    ratio_arr : jax.numpy.ndarray
        Mass ratio grid points for interpolation (q = m2/m1).
    mtot_arr : jax.numpy.ndarray
        Total mass grid points for interpolation.
    a1_arr : jax.numpy.ndarray
        Primary spin grid points for interpolation.
    a_2_arr : jax.numpy.ndarray
        Secondary spin grid points for interpolation.

    Returns
    -------
    snr : jax.numpy.ndarray
        SNR values for each detector. Shape: (n_detectors, n_samples)
    snr_effective : jax.numpy.ndarray
        Effective SNR combining all detectors. Shape: (n_samples,)
    snr_partial_ : jax.numpy.ndarray
        Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)
    d_eff : jax.numpy.ndarray
        Effective distance for each detector accounting for antenna response.
        Shape: (n_detectors, n_samples)

    Notes
    -----
    - Uses 4D cubic spline interpolation for efficient SNR calculation
    - Assumes aligned spins (no precession)
    - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
    - Chirp mass and inclination-dependent factors are computed analytically
    """
    
    det_len = detector_tensor.shape[0]
    size = mass_1.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((det_len,size))
    d_eff = jnp.zeros((det_len,size))
    snr = jnp.zeros((det_len,size))
    # loop over the detectors
    for j in range(det_len):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(spline_interp_4x4x4x4pts_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                a1_array=a1_arr,
                a2_array=a_2_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_new_batch=ratio,
                mtot_new_batch=mtot,
                a1_new_batch=a_1,
                a2_new_batch=a_2,
            ))
        )

        # calculate the effective distance
        d_eff = d_eff.at[j].set(
            luminosity_distance / jnp.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        )
        # snr[j] = snr_partial_buffer * A1 / d_eff[j]
        snr = snr.at[j].set(
            snr_partial_[j] * A1 / d_eff[j]
        )
    
    snr_effective = jnp.sqrt(jnp.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

######### Interpolation 2D #########
@jit
def spline_interp_4x4pts_jax(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new):
    """
    Function that performs the FULL 2D interpolation for a SINGLE point.
    This function finds indices, slices data, and then uses vmap internally
    to perform interpolation efficiently without Python loops.
    """

    # 1. Find indices and conditions for the new point
    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)

    # 2. Slice the small 4x4x4x4 cube of data and the corresponding 4-point coordinate arrays
    q_pts = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    m_pts = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
    F = lax.dynamic_slice(snrpartialscaled_array, (q_idx - 1, m_idx - 1), (4, 4))

    # Interpolate along the mtot dimension
    interp_along_m = vmap(
        lambda y_slice: spline_interp_4pts_jax(mtot_new, m_pts, y_slice, int_m)
    )(F)

    # Now, interpolate the resulting 4 values along the q axis
    final_snr = spline_interp_4pts_jax(q_new, q_pts, interp_along_m, int_q)

    return final_snr

# Vectorized version
@jit
def spline_interp_4x4pts_batched_jax(q_array, mtot_array, snrpartialscaled_array, q_new_batch, mtot_new_batch):
    """
    Perform batched 2D cubic spline interpolation using JAX vectorization.
    """

    # Vectorize the complete single-point interpolation function.
    # This is the only vmap call needed at the top level.
    vmapped_interpolator = vmap(
        spline_interp_4x4pts_jax,
        in_axes=(None, None, None, 0, 0)
    )

    # Call the fully vectorized function on the batch data.
    return vmapped_interpolator(
        q_array, mtot_array, snrpartialscaled_array, q_new_batch, mtot_new_batch
    )

def get_interpolated_snr_no_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000):
    """
    Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
    This function computes the SNR for gravitational wave signals with aligned spins across multiple 
    detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
    and combines results from multiple detectors to produce the effective SNR.

    Parameters
    ----------
    mass_1 : jax.numpy.ndarray
        Primary mass of the binary system in solar masses.
    mass_2 : jax.numpy.ndarray  
        Secondary mass of the binary system in solar masses.
    luminosity_distance : jax.numpy.ndarray
        Luminosity distance to the source in Mpc.
    theta_jn : jax.numpy.ndarray
        Inclination angle between the orbital angular momentum and line of sight in radians.
    psi : jax.numpy.ndarray
        Polarization angle in radians.
    geocent_time : jax.numpy.ndarray
        GPS time of coalescence at the geocenter in seconds.
    ra : jax.numpy.ndarray
        Right ascension of the source in radians.
    dec : jax.numpy.ndarray
        Declination of the source in radians.
    a_1 : jax.numpy.ndarray
        Dimensionless spin magnitude of the primary black hole.
    a_2 : jax.numpy.ndarray
        Dimensionless spin magnitude of the secondary black hole.
    detector_tensor : jax.numpy.ndarray
        Detector tensor array containing detector response information.
        Shape: (n_detectors, ...)
    snr_partialscaled : jax.numpy.ndarray
        Pre-computed scaled partial SNR values for interpolation.
        Shape: (n_detectors, ...)
    ratio_arr : jax.numpy.ndarray
        Mass ratio grid points for interpolation (q = m2/m1).
    mtot_arr : jax.numpy.ndarray
        Total mass grid points for interpolation.
    a1_arr : jax.numpy.ndarray
        Primary spin grid points for interpolation.
    a_2_arr : jax.numpy.ndarray
        Secondary spin grid points for interpolation.

    Returns
    -------
    snr : jax.numpy.ndarray
        SNR values for each detector. Shape: (n_detectors, n_samples)
    snr_effective : jax.numpy.ndarray
        Effective SNR combining all detectors. Shape: (n_samples,)
    snr_partial_ : jax.numpy.ndarray
        Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)
    d_eff : jax.numpy.ndarray
        Effective distance for each detector accounting for antenna response.
        Shape: (n_detectors, n_samples)

    Notes
    -----
    - Uses 4D cubic spline interpolation for efficient SNR calculation
    - Assumes aligned spins (no precession)
    - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
    - Chirp mass and inclination-dependent factors are computed analytically
    """

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    snr, snr_effective, snr_partial_, d_eff = get_interpolated_snr_no_spins_helper(
        jnp.array(mass_1), 
        jnp.array(mass_2), 
        jnp.array(luminosity_distance), 
        jnp.array(theta_jn), 
        jnp.array(snr_partialscaled), 
        jnp.array(ratio_arr), 
        jnp.array(mtot_arr), 
        jnp.array(Fp),
        jnp.array(Fc),
        detector_tensor
    )

    return np.array(snr), np.array(snr_effective), np.array(snr_partial_), np.array(d_eff)

@jit
def get_interpolated_snr_no_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, snr_partialscaled, ratio_arr, mtot_arr, Fp, Fc, detector_tensor):
    """
    Function to calculate the interpolated snr for a given set of parameters
    """

    size = mass_1.shape[0]
    det_len = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((det_len,size))
    d_eff = jnp.zeros((det_len,size))
    snr = jnp.zeros((det_len,size))
    # loop over the detectors
    for j in range(det_len):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(spline_interp_4x4pts_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_new_batch=ratio,
                mtot_new_batch=mtot,
            ))
        )

        d_eff = d_eff.at[j].set(
            luminosity_distance / jnp.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        )
        # snr[j] = snr_partial_buffer * A1 / d_eff[j]
        snr = snr.at[j].set(
            snr_partial_[j] * A1 / d_eff[j]
        )
    
    snr_effective = jnp.sqrt(jnp.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

# ########### For testing only ##################
# def cubic_hermite_spline_spline(p, t):
    
#     M = 0.5 * jnp.array([
#         [0,  2,  0,  0],
#         [-1, 0,  1,  0],
#         [2, -5,  4, -1],
#         [-1, 3, -3,  1]
#     ])
#     T = jnp.array([1.0, t, t**2, t**3])
#     return T @ M @ p

# @jit
# def find_index_1d(x_array, x_new):
#     N = x_array.shape[0]
#     i = jnp.sum(x_array <= x_new) - 1
#     return jnp.clip(i, 1, N - 3)

# @jit
# def cubic_hermite_spline_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
#     q_idx = find_index_1d(q_array, q_new)
#     m_idx = find_index_1d(mtot_array, mtot_new)
#     a1_idx = find_index_1d(a1_array, a1_new)
#     a2_idx = find_index_1d(a2_array, a2_new)

#     # Dynamic slices (for 4x4 neighborhood)
#     F = lax.dynamic_slice(snrpartialscaled_array, (q_idx-1, m_idx-1, a1_idx-1, a2_idx-1), (4,4,4,4))
#     qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
#     ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
#     a1s = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
#     a2s = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))

#     # Relative coordinates
#     tq = (q_new - qs[1]) / (qs[2] - qs[1])
#     tm = (mtot_new - ms[1]) / (ms[2] - ms[1])
#     ta1 = (a1_new - a1s[1]) / (a1s[2] - a1s[1])
#     ta2 = (a2_new - a2s[1]) / (a2s[2] - a2s[1])

#     # Tricubic interpolation logic, extended to 4D:
#     temp_q = jnp.zeros(4)
#     for i in range(4):
#         temp_m = jnp.zeros(4)
#         for j in range(4):
#             temp_a1 = jnp.zeros(4)
#             for k in range(4):
#                 # Interpolate along a2 (last axis)
#                 temp_a1= temp_a1.at[k].set(cubic_hermite_spline(F[i, j, k, :], ta2))
#             # Interpolate along a1
#             temp_m = temp_m.at[j].set(cubic_hermite_spline(temp_a1, ta1))
#         # Interpolate along mtot
#         temp_q = temp_q.at[i].set(cubic_hermite_spline(temp_m, tm))
#     # Interpolate along q
#     snr_new = cubic_hermite_spline(temp_q, tq)
#     return snr_new

# @jit
# def batched_cubic_hermite_spline_4d(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
#     # Vectorize only over q_new and mtot_new
#     vmapped_interp = vmap(
#         lambda q, m, a1, a2: cubic_hermite_spline_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q, m, a1, a2),
#         in_axes=(0, 0, 0, 0)
#     )
#     return vmapped_interp(q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)

# @jit
# def cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new, int_q, int_m, int_a1, int_a2):
#     """
#     Perform 4D cubic spline interpolation using JAX operations.
#     This function interpolates a 4D array (snrpartialscaled_array) at specified points
#     using cubic spline interpolation. The interpolation is performed sequentially
#     along each dimension: first a2, then a1, then mtot, and finally q.

#     Parameters
#     ----------
#     q_array : jax.numpy.ndarray
#         1D array containing the q-dimension coordinate values.
#     mtot_array : jax.numpy.ndarray
#         1D array containing the total mass dimension coordinate values.
#     a1_array : jax.numpy.ndarray
#         1D array containing the first spin parameter dimension coordinate values.
#     a2_array : jax.numpy.ndarray
#         1D array containing the second spin parameter dimension coordinate values.
#     snrpartialscaled_array : jax.numpy.ndarray
#         4D array containing the SNR partial scaled values to be interpolated.
#         Shape should be (len(q_array), len(mtot_array), len(a1_array), len(a2_array)).
#     q_new : float
#         New q value at which to interpolate.
#     mtot_new : float
#         New total mass value at which to interpolate.
#     a1_new : float
#         New first spin parameter value at which to interpolate.
#     a2_new : float
#         New second spin parameter value at which to interpolate.
#     int_q : int
#         edge condition for q interpolation. Refer to `find_index_1d_numba` for details.
#     int_m : int
#         edge condition for mtot interpolation. Refer to `find_index_1d_numba` for details.
#     int_a1 : int
#         edge condition for a1 interpolation. Refer to `find_index_1d_numba` for details.
#     int_a2 : int
#         edge condition for a2 interpolation. Refer to `find_index_1d_numba` for details.

#     Returns
#     -------
#     float
#         Interpolated SNR value at the specified (q_new, mtot_new, a1_new, a2_new) point.

#     Notes
#     -----
#     This function uses nested loops to perform interpolation sequentially along each
#     dimension. It relies on helper functions `find_index_1d_jax` for finding array
#     indices and `cubic_function_4pts_jax` for 1D cubic interpolation using 4 points.
#     The interpolation process:
#     1. Find indices and interpolation weights for each dimension
#     2. Interpolate along a2 dimension for each combination of q, mtot, a1 indices
#     3. Interpolate along a1 dimension using results from step 2
#     4. Interpolate along mtot dimension using results from step 3
#     5. Interpolate along q dimension to get the final result
#     """

#     # Find indices and conditions for q, mtot, a1, a2
#     partialsnr_along_q = jnp.zeros(4)
#     for i in range(4):
#         partialsnr_along_m = jnp.zeros(4)
#         for j in range(4):
#             partialsnr_along_a1 = jnp.zeros(4)
#             for k in range(4):
#                 # Interpolate along a2
#                 partialsnr_along_a1 = partialsnr_along_a1.at[k].set(
#                     cubic_function_4pts_jax(
#                         x_eval=a2_new, 
#                         x_pts=a2_array, 
#                         y_pts=snrpartialscaled_array[i, j, k, :], 
#                         condition_i=int_a2
#                     )
#                 )
#             # Interpolate along a1
#             partialsnr_along_m = partialsnr_along_m.at[j].set(
#                 cubic_function_4pts_jax(a1_new, a1_array, partialsnr_along_a1, int_a1)
#             )
#         # Interpolate along mtot
#         partialsnr_along_q = partialsnr_along_q.at[i].set(
#             cubic_function_4pts_jax(mtot_new, mtot_array, partialsnr_along_m, int_m)
#         )
#     # Interpolate along q
#     final_snr = cubic_function_4pts_jax(q_new, q_array, partialsnr_along_q, int_q)
    
#     return final_snr

# def _helper(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
#     """
#     Processes a single set of parameters. This function is designed to be vectorized by vmap.
#     All array arguments are passed explicitly.
#     """
#     # Find the index for each parameter for a single item
#     q_idx, q_condition_i = find_index_1d_jax(q_array, q_new)
#     m_idx, m_condition_i = find_index_1d_jax(mtot_array, mtot_new)
#     a1_idx, a1_condition_i = find_index_1d_jax(a1_array, a1_new)
#     a2_idx, a2_condition_i = find_index_1d_jax(a2_array, a2_new)

#     # Slice the 1D arrays to get the 4 points around the index
#     q_array_slice = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
#     m_array_slice = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
#     a1_array_slice = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
#     a2_array_slice = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))

#     # Slice the main 4D data cube
#     cube_4x4x4x4 = lax.dynamic_slice(
#         snrpartialscaled_array,
#         (q_idx - 1, m_idx - 1, a1_idx - 1, a2_idx - 1),
#         (4, 4, 4, 4)
#     )

#     # Return all results for the single item
#     return (q_idx, q_condition_i, m_idx, m_condition_i, a1_idx, a1_condition_i, a2_idx, a2_condition_i,
#             q_array_slice, m_array_slice, a1_array_slice, a2_array_slice, cube_4x4x4x4)

# # The main function is JIT-compiled once.
# @jit
# def cubic_spline_4d_batched_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
#     """
#     Perform batched 4D cubic spline interpolation using JAX vectorization.
#     """
#     # Vectorize the helper function directly.
#     # Use in_axes to specify that the first 5 arguments are static (None)
#     # and the next 4 are batched over the first axis (0).
#     vmapped_helper = vmap(
#         _helper,
#         in_axes=(None, None, None, None, None, 0, 0, 0, 0)
#     )

#     # Call the vectorized function.
#     # The static arrays are passed once and are broadcasted by vmap, not iterated over.
#     q_idx, q_condition_i, m_idx, m_condition_i, a1_idx, a1_condition_i, a2_idx, a2_condition_i, q_array_slice, m_array_slice, a1_array_slice, a2_array_slice, cube_4x4x4x4 = vmapped_helper(
#         q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array,
#         q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch
#     )

#     results = vmap(cubic_spline_4d_jax)(
#         q_array_slice, m_array_slice, a1_array_slice, a2_array_slice, cube_4x4x4x4,
#         q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch
#     )

#     return results


# #################