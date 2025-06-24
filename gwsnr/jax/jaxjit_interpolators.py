# -*- coding: utf-8 -*-

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, lax
from .jaxjit_functions import antenna_response_array

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
def cubic_function_4pts_jax(x_eval, x_pts, y_pts, condition_i):
    """
    Performs piecewise interpolation using 4 points with JAX compatibility.
    This function implements a piecewise interpolation scheme that uses:
    - Linear interpolation at the left boundary (condition_i=1)
    - Cubic interpolation in the middle region (condition_i=2)
    - Linear interpolation at the right boundary (condition_i=3)
    The cubic interpolation uses Catmull-Rom spline coefficients for smooth
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
    def cubic_branch(operands):
        x_eval, x_pts, y_pts = operands
        x0, x1, x2, x3 = x_pts[0], x_pts[1], x_pts[2], x_pts[3]
        y0, y1, y2, y3 = y_pts[0], y_pts[1], y_pts[2], y_pts[3]
        denom = x2 - x1
        # fallback to y1 if spacing is degenerate
        # jnp.where(denom == 0.0, 1.0, denom)
        t = (x_eval - x1) / denom
        a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
        b = y0 - 2.5*y1 + 2.0*y2 - 0.5*y3
        c = -0.5*y0 + 0.5*y2
        d = y1

        return a*t**3 + b*t**2 + c*t + d
    
    return jax.lax.switch(
        condition_i - 1,  # 1 -> 0, 2 -> 1, 3 -> 2
        [left_branch, cubic_branch, right_branch],
        (x_eval, x_pts, y_pts)
    )

######### Interpolation 4D #########
@jit
def cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
    """
    Perform 4D cubic spline interpolation using JAX operations.
    This function interpolates a 4D array (snrpartialscaled_array) at specified points
    using cubic spline interpolation. The interpolation is performed sequentially
    along each dimension: first a2, then a1, then mtot, and finally q.

    Parameters
    ----------
    q_array : jax.numpy.ndarray
        1D array containing the q-dimension coordinate values.
    mtot_array : jax.numpy.ndarray
        1D array containing the total mass dimension coordinate values.
    a1_array : jax.numpy.ndarray
        1D array containing the first spin parameter dimension coordinate values.
    a2_array : jax.numpy.ndarray
        1D array containing the second spin parameter dimension coordinate values.
    snrpartialscaled_array : jax.numpy.ndarray
        4D array containing the SNR partial scaled values to be interpolated.
        Shape should be (len(q_array), len(mtot_array), len(a1_array), len(a2_array)).
    q_new : float
        New q value at which to interpolate.
    mtot_new : float
        New total mass value at which to interpolate.
    a1_new : float
        New first spin parameter value at which to interpolate.
    a2_new : float
        New second spin parameter value at which to interpolate.

    Returns
    -------
    float
        Interpolated SNR value at the specified (q_new, mtot_new, a1_new, a2_new) point.

    Notes
    -----
    This function uses nested loops to perform interpolation sequentially along each
    dimension. It relies on helper functions `find_index_1d_jax` for finding array
    indices and `cubic_function_4pts_jax` for 1D cubic interpolation using 4 points.
    The interpolation process:
    1. Find indices and interpolation weights for each dimension
    2. Interpolate along a2 dimension for each combination of q, mtot, a1 indices
    3. Interpolate along a1 dimension using results from step 2
    4. Interpolate along mtot dimension using results from step 3
    5. Interpolate along q dimension to get the final result
    """

    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)
    a1_idx, int_a1 = find_index_1d_jax(a1_array, a1_new)
    a2_idx, int_a2 = find_index_1d_jax(a2_array, a2_new)

    F = lax.dynamic_slice(snrpartialscaled_array, (q_idx - 1, m_idx - 1, a1_idx - 1, a2_idx - 1), (4, 4, 4, 4))
    qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
    a1s = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
    a2s = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))

    vals_after_a1 = jnp.zeros(4)
    for i in range(4):
        vals_after_m = jnp.zeros(4)
        for j in range(4):
            vals_after_a2 = jnp.zeros(4)
            for k in range(4):
                # Interpolate along a2
                vals_after_a2 = vals_after_a2.at[k].set(
                    cubic_function_4pts_jax(a2_new, a2s, F[i, j, k, :], int_a2)
                )
            # Interpolate along a1
            vals_after_m = vals_after_m.at[j].set(
                cubic_function_4pts_jax(a1_new, a1s, vals_after_a2, int_a1)
            )
        # Interpolate along mtot
        vals_after_a1 = vals_after_a1.at[i].set(
            cubic_function_4pts_jax(mtot_new, ms, vals_after_m, int_m)
        )
    # Interpolate along q
    final_snr = cubic_function_4pts_jax(q_new, qs, vals_after_a1, int_q)
    return final_snr

# Vectorized version
@jit
def cubic_spline_4d_batched_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_batch, mtot_batch, a1_batch, a2_batch):
    """
    Perform batched 4D cubic spline interpolation using JAX vectorization.
    This function applies 4D cubic spline interpolation to batches of input parameters
    using JAX's vmap for efficient vectorized computation. It interpolates SNR values
    based on mass ratio (q), total mass (mtot), and two spin parameters (a1, a2).

    Parameters
    ----------
    q_array : jax.numpy.ndarray
        1D array of mass ratio grid points for interpolation.
    mtot_array : jax.numpy.ndarray
        1D array of total mass grid points for interpolation.
    a1_array : jax.numpy.ndarray
        1D array of first spin parameter grid points for interpolation.
    a2_array : jax.numpy.ndarray
        1D array of second spin parameter grid points for interpolation.
    snrpartialscaled_array : jax.numpy.ndarray
        4D array of SNR values corresponding to the grid points, with shape
        (len(q_array), len(mtot_array), len(a1_array), len(a2_array)).
    q_batch : jax.numpy.ndarray
        1D array of mass ratio values to interpolate at.
    mtot_batch : jax.numpy.ndarray
        1D array of total mass values to interpolate at.
    a1_batch : jax.numpy.ndarray
        1D array of first spin parameter values to interpolate at.
    a2_batch : jax.numpy.ndarray
        1D array of second spin parameter values to interpolate at.

    Returns
    -------
    jax.numpy.ndarray
        1D array of interpolated SNR values with the same length as the input batches.

    Notes
    -----
    - All batch arrays must have the same length.
    - Uses JAX's vmap for efficient vectorized computation.
    - Calls cubic_spline_4d_jax internally for each set of parameters.
    """

    # vmapped_interp = vmap(lambda q, m: spline_4d_interp_single(q_array, mtot_array, snrpartialscaled_array, q, m), in_axes=(0, 0))
    vmapped_interp = vmap(lambda q, m, a1, a2: cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q, m, a1, a2), in_axes=(0, 0, 0, 0))
    return vmapped_interp(q_batch, mtot_batch, a1_batch, a2_batch)

@jit
def get_interpolated_snr_aligned_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr):
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
    
    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((len_,size))
    d_eff = jnp.zeros((len_,size))
    snr = jnp.zeros((len_,size))
    # loop over the detectors
    for j in range(len_):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(cubic_spline_4d_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                a1_array=a1_arr,
                a2_array=a_2_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_batch=ratio,
                mtot_batch=mtot,
                a1_batch=a_1,
                a2_batch= a_2,
            ))
        )

        # calculate the effective distance
        # d_eff[j] =luminosity_distance / jnp.sqrt(
        #             Fp[j]**2 * ci_param + Fc[j]**2 * ci_2
        #         )
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
def cubic_spline_2d_jax(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new):
    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)

    F = lax.dynamic_slice(snrpartialscaled_array, (q_idx - 1, m_idx - 1), (4, 4))
    qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))

    partialsnr_q = jnp.zeros(4)
    partialsnr_q = jnp.zeros(4)
    for i in range(4):
        partialsnr_m = jnp.zeros(4)
        # Interpolate along mtot
        partialsnr_q = partialsnr_q.at[i].set(cubic_function_4pts_jax(mtot_new, ms, F[i, :], int_m))
    # Interpolate along q
    snr_final = cubic_function_4pts_jax(q_new, qs, partialsnr_q, int_q)
    return snr_final

# Vectorized version
@jit
def cubic_spline_2d_batched_jax(q_array, mtot_array, snrpartialscaled_array, q_batch, mtot_batch):
    vmapped_interp = vmap(lambda q, m: cubic_spline_2d_jax(q_array, mtot_array, snrpartialscaled_array, q, m), in_axes=(0, 0))
    return vmapped_interp(q_batch, mtot_batch)

@jit
def get_interpolated_snr_no_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr):
    """
    Function to calculate the in terpolated snr for a given set of parameters
    """

    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((len_,size))
    d_eff = jnp.zeros((len_,size))
    snr = jnp.zeros((len_,size))
    # loop over the detectors
    for j in range(len_):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(cubic_spline_2d_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_batch=ratio,
                mtot_batch=mtot,
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
# def catmull_rom_spline(p, t):
    
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
# def catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
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
#                 temp_a1= temp_a1.at[k].set(catmull_rom_spline(F[i, j, k, :], ta2))
#             # Interpolate along a1
#             temp_m = temp_m.at[j].set(catmull_rom_spline(temp_a1, ta1))
#         # Interpolate along mtot
#         temp_q = temp_q.at[i].set(catmull_rom_spline(temp_m, tm))
#     # Interpolate along q
#     snr_new = catmull_rom_spline(temp_q, tq)
#     return snr_new

# @jit
# def batched_catmull_rom_4d(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
#     # Vectorize only over q_new and mtot_new
#     vmapped_interp = vmap(
#         lambda q, m, a1, a2: catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q, m, a1, a2),
#         in_axes=(0, 0, 0, 0)
#     )
#     return vmapped_interp(q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)


# #################