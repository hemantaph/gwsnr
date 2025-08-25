import mlx.core as mx
import numpy as np
from ..numba import antenna_response_array

@mx.compile
def find_index_1d_mlx(x_array, x_new):
    N = x_array.shape[0]
    i = mx.sum(x_array <= x_new) - 1
    i = mx.clip(i, 1, N - 3)
    condition_i = mx.where(x_new <= x_array[1], 1,
                  mx.where(x_new >= x_array[N-2], 3, 2))
    return i, condition_i

@mx.compile
def spline_interp_4pts_mlx(x_eval, x_pts, y_pts, condition_i):
    
    # Linear at boundaries
    def left_branch(x_eval, x_pts, y_pts):
        denom_left = x_pts[1] - x_pts[0]
        left_val = y_pts[0] + (y_pts[1] - y_pts[0]) * (x_eval - x_pts[0]) / denom_left
        return left_val

    def right_branch(x_eval, x_pts, y_pts):
        denom_right = x_pts[3] - x_pts[2]
        right_val = y_pts[2] + (y_pts[3] - y_pts[2]) * (x_eval - x_pts[2]) / denom_right
        return right_val

    # Cubic in middle
    def cubic_branch(x_eval, x_pts, y_pts):
        x0, x1, x2, x3 = x_pts[0], x_pts[1], x_pts[2], x_pts[3]
        y0, y1, y2, y3 = y_pts[0], y_pts[1], y_pts[2], y_pts[3]
        denom = x2 - x1
        t = (x_eval - x1) / denom

        m1 = ((y2 - y1) / (x2 - x1)) * ((x1 - x0) / (x2 - x0)) + ((y1 - y0) / (x1 - x0)) * ((x2 - x1) / (x2 - x0))
        m2 = ((y3 - y2) / (x3 - x2)) * ((x2 - x1) / (x3 - x1)) + ((y2 - y1) / (x2 - x1)) * ((x3 - x2) / (x3 - x1))

        h00 = (2 * t**3 - 3 * t**2 + 1)
        h10 = (t**3 - 2 * t**2 + t)
        h01 = (-2 * t**3 + 3 * t**2)
        h11 = (t**3 - t**2)

        cubic_val = h00 * y1 + h10 * m1 * denom + h01 * y2 + h11 * m2 * denom
        return cubic_val

    # Use chained mx.where for multi-way select
    return mx.where(condition_i == 1, left_branch(x_eval, x_pts, y_pts),
                   mx.where(condition_i == 3, right_branch(x_eval, x_pts, y_pts), cubic_branch(x_eval, x_pts, y_pts)))

######### Interpolation 4D #########
@mx.compile
def spline_interp_4x4x4x4pts_mlx(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
    """
    Helper function that performs the FULL 4D interpolation for a SINGLE point.
    This function finds indices, slices data, and then uses vmap internally
    to perform interpolation efficiently without Python loops.
    """
    # 1. Find indices and conditions for the new point
    q_idx, int_q = find_index_1d_mlx(q_array, q_new)
    m_idx, int_m = find_index_1d_mlx(mtot_array, mtot_new)
    a1_idx, int_a1 = find_index_1d_mlx(a1_array, a1_new)
    a2_idx, int_a2 = find_index_1d_mlx(a2_array, a2_new)

    # 2. Slice the small 4x4x4x4 cube of data and the corresponding 4-point coordinate arrays
    q_pts = mx.array([q_array[q_idx - 1], q_array[q_idx], q_array[q_idx + 1], q_array[q_idx + 2]])
    m_pts = mx.array([mtot_array[m_idx - 1], mtot_array[m_idx], mtot_array[m_idx + 1], mtot_array[m_idx + 2]])
    a1_pts = mx.array([a1_array[a1_idx - 1], a1_array[a1_idx], a1_array[a1_idx + 1], a1_array[a1_idx + 2]])
    a2_pts = mx.array([a2_array[a2_idx - 1], a2_array[a2_idx], a2_array[a2_idx + 1], a2_array[a2_idx + 2]])

    # # --- Create the 4x4x4x4 data_cube ---
    # # We expand the index arrays with new dimensions so they broadcast correctly
    # # to select a 4x4x4x4 cube from the larger array.
    offsets = mx.arange(-1, 3)
    q_indices  = (q_idx + offsets).reshape(4, 1, 1, 1)
    m_indices  = (m_idx + offsets).reshape(1, 4, 1, 1)
    a1_indices = (a1_idx + offsets).reshape(1, 1, 4, 1)
    a2_indices = (a2_idx + offsets).reshape(1, 1, 1, 4)

    # Perform a single, efficient "gather" operation using the broadcasted indices.
    data_cube = snrpartialscaled_array[q_indices, m_indices, a1_indices, a2_indices]


    # 3. Perform 4D interpolation using vmap to eliminate loops
    # Interpolate along the last dimension (a2) for each of the 4x4x4 slices
    interp_on_a2 = mx.vmap(
        lambda y_slice: spline_interp_4pts_mlx(a2_new, a2_pts, y_slice, int_a2)
    )(data_cube.reshape(64, 4)).reshape(4, 4, 4)

    # Interpolate along the next dimension (a1)
    interp_on_a1 = mx.vmap(
        lambda y_slice: spline_interp_4pts_mlx(a1_new, a1_pts, y_slice, int_a1)
    )(interp_on_a2.reshape(16, 4)).reshape(4, 4)

    # Interpolate along the mtot dimension
    interp_on_m = mx.vmap(
        lambda y_slice: spline_interp_4pts_mlx(mtot_new, m_pts, y_slice, int_m)
    )(interp_on_a1)

    # Final interpolation along the q dimension
    final_snr = spline_interp_4pts_mlx(q_new, q_pts, interp_on_m, int_q)

    return final_snr

@mx.compile
def spline_interp_4x4x4x4pts_batched_mlx(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
    """
    Perform batched 4D cubic spline interpolation using JAX vectorization.
    """
    # Vectorize the complete single-point interpolation function.
    # This is the only vmap call needed at the top level.
    vmapped_interpolator = mx.vmap(
        spline_interp_4x4x4x4pts_mlx,
        in_axes=(None, None, None, None, None, 0, 0, 0, 0)
    )

    # Call the fully vectorized function on the batch data.
    return vmapped_interpolator(
        q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array,
        q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch
    )

def get_interpolated_snr_aligned_spins_mlx(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000):
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
    # antenna_response_array is already compiled with numba
    # this is done so as to preserve the float64 precesion in the antenna response calculation
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    snr, snr_effective, snr_partial_, d_eff = get_interpolated_snr_aligned_spins_helper(
        mx.array(mass_1), 
        mx.array(mass_2), 
        mx.array(luminosity_distance), 
        mx.array(theta_jn), 
        mx.array(a_1), 
        mx.array(a_2), 
        mx.array(snr_partialscaled), 
        mx.array(ratio_arr), 
        mx.array(mtot_arr), 
        mx.array(a1_arr), 
        mx.array(a_2_arr),
        mx.array(Fp),
        mx.array(Fc),
        mx.array(detector_tensor),
        batch_size
    )

    return np.array(snr), np.array(snr_effective), np.array(snr_partial_), np.array(d_eff)

@mx.compile
def get_interpolated_snr_aligned_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, a_1, a_2, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, Fp, Fc, detector_tensor, batch_size):
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
    det_len = detector_tensor.shape[0]
    
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = mx.cos(theta_jn) ** 2
    ci_param = ((1 + mx.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = mx.zeros((det_len,size))
    d_eff = mx.zeros((det_len,size))
    snr = mx.zeros((det_len,size))
    # loop over the detectors
    for j in range(det_len):
        # Iterate over the data in batches
        for i in range(0, size, batch_size):
            # Define the start and end indices for the current batch
            start_idx = i
            end_idx = min(i + batch_size, size)

            snr_partial_[j, start_idx:end_idx] = mx.array(spline_interp_4x4x4x4pts_batched_mlx(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                a1_array=a1_arr,
                a2_array=a_2_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_new_batch=ratio,
                mtot_new_batch=mtot,
                a1_new_batch=a_1,
                a2_new_batch=a_2,
            )
        )

        d_eff[j] = luminosity_distance / mx.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        # Calculate the SNR for this detector
        snr[j] = snr_partial_[j] * A1 / d_eff[j]

    snr_effective = mx.sqrt(mx.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff


######### Interpolation 2D #########
@mx.compile
def spline_interp_4x4pts_mlx(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new):
    """
    Helper function that performs the FULL 2D interpolation for a SINGLE point.
    This function finds indices, slices data, and then uses vmap internally
    to perform interpolation efficiently without Python loops.
    """
    # 1. Find indices and conditions for the new point
    q_idx, int_q = find_index_1d_mlx(q_array, q_new)
    m_idx, int_m = find_index_1d_mlx(mtot_array, mtot_new)

    # 2. Slice the small 4x4 cube of data and the corresponding 4-point coordinate arrays
    q_pts = mx.array([q_array[q_idx - 1], q_array[q_idx], q_array[q_idx + 1], q_array[q_idx + 2]])
    m_pts = mx.array([mtot_array[m_idx - 1], mtot_array[m_idx], mtot_array[m_idx + 1], mtot_array[m_idx + 2]])

    # # --- Create the 4x4 data_cube ---
    # # We expand the index arrays with new dimensions so they broadcast correctly
    # # to select a 4x4 cube from the larger array.
    offsets = mx.arange(-1, 3)
    q_indices  = (q_idx + offsets).reshape(4, 1)
    m_indices  = (m_idx + offsets).reshape(1, 4)

    # Perform a single, efficient "gather" operation using the broadcasted indices.
    data_cube = snrpartialscaled_array[q_indices, m_indices]

    # 3. Perform 2D interpolation using vmap to eliminate loops
    # Interpolate along the mtot dimension
    interp_on_m = mx.vmap(
        lambda y_slice: spline_interp_4pts_mlx(mtot_new, m_pts, y_slice, int_m)
    )(data_cube)

    # Final interpolation along the q dimension
    final_snr = spline_interp_4pts_mlx(q_new, q_pts, interp_on_m, int_q)

    return final_snr

@mx.compile
def spline_interp_4x4pts_batched_mlx(q_array, mtot_array, snrpartialscaled_array, q_new_batch, mtot_new_batch):
    """
    Perform batched 2D cubic spline interpolation using MLX vectorization.
    """
    # Vectorize the complete single-point interpolation function.
    # This is the only vmap call needed at the top level.
    vmapped_interpolator = mx.vmap(
        spline_interp_4x4pts_mlx,
        in_axes=(None, None, None, 0, 0)
    )

    # Call the fully vectorized function on the batch data.
    return vmapped_interpolator(
        q_array, mtot_array, snrpartialscaled_array,
        q_new_batch, mtot_new_batch
    )

def get_interpolated_snr_no_spins_mlx(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000):
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
    # antenna_response_array is already compiled with numba
    # this is done so as to preserve the float64 precesion in the antenna response calculation
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    snr, snr_effective, snr_partial_, d_eff = get_interpolated_snr_no_spins_helper(
        mx.array(mass_1), 
        mx.array(mass_2), 
        mx.array(luminosity_distance), 
        mx.array(theta_jn), 
        mx.array(snr_partialscaled), 
        mx.array(ratio_arr), 
        mx.array(mtot_arr), 
        mx.array(Fp),
        mx.array(Fc),
        mx.array(detector_tensor),
        batch_size
    )

    return np.array(snr), np.array(snr_effective), np.array(snr_partial_), np.array(d_eff)

@mx.compile
def get_interpolated_snr_no_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, snr_partialscaled, ratio_arr, mtot_arr, Fp, Fc, detector_tensor, batch_size):
    """
    Calculate interpolated signal-to-noise ratio (SNR) for non-spinning gravitational wave signals using JAX.
    This function computes the SNR for gravitational wave signals without spins across multiple 
    detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
    and combines results from multiple detectors to produce the effective SNR.
    """
    
    size = mass_1.shape[0]
    det_len = detector_tensor.shape[0]
    
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = mx.cos(theta_jn) ** 2
    ci_param = ((1 + mx.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = mx.zeros((det_len,size))
    d_eff = mx.zeros((det_len,size))
    snr = mx.zeros((det_len,size))
    # loop over the detectors
    for j in range(det_len):
        # Iterate over the data in batches
        for i in range(0, size, batch_size):
            # Define the start and end indices for the current batch
            start_idx = i
            end_idx = min(i + batch_size, size)
            
            snr_partial_[j, start_idx:end_idx] = mx.array(spline_interp_4x4pts_batched_mlx(
                    q_array=ratio_arr,
                    mtot_array=mtot_arr,
                    snrpartialscaled_array=snr_partialscaled[j],
                    q_new_batch=ratio,
                    mtot_new_batch=mtot
                )
            )

        d_eff[j] = luminosity_distance / mx.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        # Calculate the SNR for this detector
        snr[j] = snr_partial_[j] * A1 / d_eff[j]

    snr_effective = mx.sqrt(mx.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff