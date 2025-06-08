import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from .njit_functions import (
    antenna_response_array,
)


@jit
def findchirp_chirptime_jax(m1, m2, fmin):
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

    Gamma = 0.5772156649015329
    Pi = jnp.pi
    MTSUN_SI = 4.925491025543576e-06
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
        + jnp.log(4.0) * 6848.0 / 105.0
    )
    c6LogT = 6848.0 / 105.0

    c5T = 13.0 * Pi * eta / 3.0 - 7729.0 * Pi / 252.0

    c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0)
    c3T = -32.0 * Pi / 5.0
    c2T = 743.0 / 252.0 + eta * 11.0 / 3.0
    c0T = 5.0 * m * MTSUN_SI / (256.0 * eta)

    # This is the PN parameter v evaluated at the lower freq. cutoff
    xT = jnp.power(Pi * m * MTSUN_SI * fmin, 1.0 / 3.0)
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
            + (c6T + c6LogT * jnp.log(xT)) * x6T
            + c7T * x7T
        )
        / x8T
    )


@jit
def catmull_rom_spline(p, t):
    M = 0.5 * jnp.array([
        [0,  2,  0,  0],
        [-1, 0,  1,  0],
        [2, -5,  4, -1],
        [-1, 3, -3,  1]
    ])
    T = jnp.array([1.0, t, t**2, t**3])
    return T @ M @ p

@jit
def find_index_1d(x_array, x_new):
    N = x_array.shape[0]
    i = jnp.sum(x_array <= x_new) - 1
    return jnp.clip(i, 1, N - 3)

@jit
def catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snr_array, q_new, mtot_new, a1_new, a2_new):
    q_idx = find_index_1d(q_array, q_new)
    m_idx = find_index_1d(mtot_array, mtot_new)
    a1_idx = find_index_1d(a1_array, a1_new)
    a2_idx = find_index_1d(a2_array, a2_new)

    # Dynamic slices (for 4x4 neighborhood)
    F = lax.dynamic_slice(snr_array, (q_idx-1, m_idx-1, a1_idx-1, a2_idx-1), (4,4,4,4))
    qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
    a1s = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
    a2s = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))

    # Relative coordinates
    tq = (q_new - qs[1]) / (qs[2] - qs[1])
    tm = (mtot_new - ms[1]) / (ms[2] - ms[1])
    ta1 = (a1_new - a1s[1]) / (a1s[2] - a1s[1])
    ta2 = (a2_new - a2s[1]) / (a2s[2] - a2s[1])

    # Tricubic interpolation logic, extended to 4D:
    temp_q = jnp.zeros(4)
    for i in range(4):
        temp_m = jnp.zeros(4)
        for j in range(4):
            temp_a1 = jnp.zeros(4)
            for k in range(4):
                # Interpolate along a2 (last axis)
                temp_a1= temp_a1.at[k].set(catmull_rom_spline(F[i, j, k, :], ta2))
            # Interpolate along a1
            temp_m = temp_m.at[j].set(catmull_rom_spline(temp_a1, ta1))
        # Interpolate along mtot
        temp_q = temp_q.at[i].set(catmull_rom_spline(temp_m, tm))
    # Interpolate along q
    snr_new = catmull_rom_spline(temp_q, tq)
    return snr_new

@jit
def batched_catmull_rom_4d(q_array, mtot_array, a1_array, a2_array, snr_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
    # Vectorize only over q_new and mtot_new
    vmapped_interp = vmap(
        lambda q, m, a1, a2: catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snr_array, q, m, a1, a2),
        in_axes=(0, 0, 0, 0)
    )
    return vmapped_interp(q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)


def get_interpolated_snr_aligned_spins(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr):
    """
    Function to calculate the interpolated snr for a given set of parameters

    Parameters
    ----------
    mass_1 : `numpy.ndarray`
        Mass of the first body in solar masses.
    mass_2 : `numpy.ndarray`
        Mass of the second body in solar masses.
    luminosity_distance : `float`
        Luminosity distance to the source in Mpc.
    theta_jn : `numpy.ndarray`
        Angle between the total angular momentum and the line of sight to the source in radians.
    psi : `numpy.ndarray`
        Polarization angle of the source.
    geocent_time : `numpy.ndarray`
        GPS time of the source.
    ra : ``numpy.ndarray`
        Right ascension of the source in radians.
    dec : `numpy.ndarray`
        Declination of the source in radians.
    detector_tensor : array-like
        Detector tensor for the detector (3x3 matrix)
    snr_partialscaled : `numpy.ndarray`
        Array of snr_partialscaled coefficients for the detector.
    ratio_arr : `numpy.ndarray`
        Array of mass ratio values for the snr_partialscaled coefficients.
    mtot_arr : `numpy.ndarray`
        Array of total mass values for the snr_partialscaled coefficients.
    
    Returns
    -------
    snr : `float`
        snr of the detector.
    """

    size = len(mass_1)
    len_ = len(detector_tensor)
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1
    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2
    
    size = len(mass_1)
    snr_partial_ = np.zeros((len_,size))
    d_eff = np.zeros((len_,size))
    snr = np.zeros((len_,size))
    # loop over the detectors
    for j in range(len_):
        snr_partial_buffer = np.array(batched_catmull_rom_4d(
            q_array=jnp.array(ratio_arr),
            mtot_array=jnp.array(mtot_arr),
            a1_array=jnp.array(a1_arr),
            a2_array=jnp.array(a_2_arr),
            snr_array=jnp.array(snr_partialscaled[j]),
            q_new_batch=jnp.array(ratio),
            mtot_new_batch=jnp.array(mtot),
            a1_new_batch=jnp.array(a_1),
            a2_new_batch= jnp.array(a_2),
        ), dtype=float)

        # calculate the effective distance
        d_eff[j] =luminosity_distance / np.sqrt(
                    Fp[j]**2 * ci_param + Fc[j]**2 * ci_2
                )
        snr[j] = snr_partial_buffer * A1 / d_eff[j]
    
    snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff