import numpy as np
from numba import njit, prange
from .njit_functions import antenna_response_array


@njit
def find_index_1d_numba(x_array, x_new):
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
    i : int
        The index in `x_array` such that `x_array[i-1] <= x_new < x_array[i+1]`.
    condition_i : int
        An integer indicating the condition for edge handling:
        - 1: `x_new` is less than or equal to the first element of `x_array`.
        - 2: `x_new` is between the first and last elements of `x_array`.
        - 3: `x_new` is greater than or equal to the last element of `x_array`.
    
    Notes
    -----
    Uses binary search with clipped indices to ensure valid 4-point stencils.
    The condition parameter determines linear vs cubic interpolation at boundaries.
    """

    N = x_array.shape[0]
    # Equivalent of: i = np.sum(x_array <= x_new) - 1
    i = np.searchsorted(x_array, x_new, side='right') - 1
    # Clip to 1, N-3
    if i < 1:
        i = 1
    elif i > N - 3:
        i = N - 3

    condition_i = 2  # default
    if (x_new <= x_array[1]):
        condition_i = 1
    elif (x_new >= x_array[N-2]):
        condition_i = 3

    return i, condition_i
    
@njit
def spline_interp_4pts_numba(x_eval, x_pts, y_pts, condition_i):
    """
    Evaluate a cubic function at a given point using 4-point interpolation.
    
    Parameters
    ----------
    x_eval : float
        The x value at which to evaluate the cubic function.
    x_pts : jnp.ndarray
        The x values of the 4 points used for interpolation. Must be sorted in ascending order
    y_pts : jnp.ndarray
        The y values corresponding to the x_pts. Must have the same length as x_pts.
    condition_i : int
        An integer indicating the condition for edge handling:
        - 1: `x_eval` is less than or equal to the first element of `x_pts`.
        - 2: `x_eval` is between the first and last elements of `x_pts`.
        - 3: `x_eval` is greater than or equal to the last element of `x_pts`.
        
    Returns
    -------
    float
        The interpolated value at `x_eval`.

    Notes
    -----
    This function uses cubic Hermite interpolation for the main case (condition_i == 2).
    For edge cases (condition_i == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.
    The x_pts and y_pts must be of length 4, and x_pts must be sorted in ascending order.
    The function assumes that the input arrays are valid and does not perform additional checks.
    If the input arrays are not of length 4, or if x_pts is not sorted, the behavior is undefined.
    The function is designed to be used with Numba's JIT compilation for performance.
    It is optimized for speed and does not include error handling or input validation.
    The cubic Hermite interpolation is based on the tangents calculated from the y values at the four points.
    The tangents are computed using the differences between the y values and the x values
    to ensure smoothness and continuity of the interpolated curve.
    """
    # Handle edges with linear interpolation
    if condition_i == 2:
        # Main Hermite cubic spline, between x_pts[1] and x_pts[2]
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
    elif condition_i == 1:
        return y_pts[0] + (y_pts[1] - y_pts[0]) * (x_eval - x_pts[0]) / (x_pts[1] - x_pts[0])
    elif condition_i == 3:
        return y_pts[2] + (y_pts[3] - y_pts[2]) * (x_eval - x_pts[2]) / (x_pts[3] - x_pts[2])
    
######### Interpolation 4D #########
@njit
def spline_interp_4x4x4x4pts_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new, int_q, int_m, int_a1, int_a2):
    """
    Perform cubic spline interpolation in 4D for the given arrays and new values.
    
    Parameters
    ----------
    q_array : jnp.ndarray
        The array of q values for interpolation. Must be sorted in ascending order.
    mtot_array : jnp.ndarray
        The array of mtot values for interpolation. Must be sorted in ascending order.
    a1_array : jnp.ndarray
        The array of a1 values for interpolation. Must be sorted in ascending order.
    a2_array : jnp.ndarray
        The array of a2 values for interpolation. Must be sorted in ascending order.
    snrpartialscaled_array : jnp.ndarray
        The 4D array of snrpartialscaled values with shape (4, 4, 4, 4).
        This array contains the values to be interpolated.
    q_new : float
        The new q value at which to evaluate the cubic spline.
    mtot_new : float
        The new mtot value at which to evaluate the cubic spline.
    a1_new : float
        The new a1 value at which to evaluate the cubic spline.
    a2_new : float
        The new a2 value at which to evaluate the cubic spline.
    int_q : int
        edge condition for q interpolation. Refer to `find_index_1d_numba` for details.
    int_m : int
        edge condition for mtot interpolation. Refer to `find_index_1d_numba` for details.
    int_a1 : int
        edge condition for a1 interpolation. Refer to `find_index_1d_numba` for details.
    int_a2 : int
        edge condition for a2 interpolation. Refer to `find_index_1d_numba` for details.
        
    Returns
    -------
    float
        The interpolated value at the new coordinates (q_new, mtot_new, a1_new, a2_new).

    Notes
    -----
    This function uses cubic Hermite interpolation for the main case (int_q == 2, int_m == 2, int_a1 == 2, int_a2 == 2).
    For edge cases (int_q == 1 or 3, int_m == 1 or 3, int_a1 == 1 or 3, int_a2 == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.
    """

    # Find indices and conditions for q, mtot, a1, a2
    partialsnr_along_q = np.zeros(4)
    for i in range(4): # Loop over q-dimension index offset
        partialsnr_along_m = np.zeros(4)
        for j in range(4): # Loop over mtot-dimension index offset
            partialsnr_along_a1 = np.zeros(4)
            for k in range(4): # Loop over a1-dimension index offset
                # Get the 4 y-points for a2 interpolation directly from the main array
                # y_pts_a2 = snrpartialscaled_array[q_idx-1+i, m_idx-1+j, a1_idx-1+k, a2_idx-1:a2_idx+3]
                # Numba doesn't like fancy indexing, so build array manually:
                # partialsnr_along_a2 = snrpartialscaled_array[i, j, k, :]

                # Interpolate for a2
                # find partialsnr at a2_new, along a2 axis, for fixed a1
                partialsnr_along_a1[k] = spline_interp_4pts_numba(
                    x_eval=a2_new, 
                    x_pts=a2_array, 
                    y_pts=snrpartialscaled_array[i, j, k, :], 
                    condition_i=int_a2
                )

            partialsnr_along_m[j] = spline_interp_4pts_numba(
                a1_new,
                a1_array,
                partialsnr_along_a1,
                int_a1
            )

        partialsnr_along_q[i] = spline_interp_4pts_numba(
            mtot_new, 
            mtot_array, 
            partialsnr_along_m, 
            int_m
        )

    final_snr = spline_interp_4pts_numba(
        q_new,
        q_array,
        partialsnr_along_q,
        int_q
    )
    
    return final_snr

@njit(parallel=True)
def spline_interp_4x4x4x4pts_batched_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
    """
    Perform cubic spline interpolation in 4D for a batch of new values.

    Parameters
    ----------
    q_array : jnp.ndarray
        The array of q values for interpolation. Must be sorted in ascending order.
    mtot_array : jnp.ndarray
        The array of mtot values for interpolation. Must be sorted in ascending order.
    a1_array : jnp.ndarray
        The array of a1 values for interpolation. Must be sorted in ascending order.
    a2_array : jnp.ndarray
        The array of a2 values for interpolation. Must be sorted in ascending order.
    snrpartialscaled_array : jnp.ndarray
        The 4D array of snrpartialscaled values.
    q_new_batch : jnp.ndarray
        The new q values at which to evaluate the cubic spline. Must be a 1D array.
    mtot_new_batch : jnp.ndarray
        The new mtot values at which to evaluate the cubic spline. Must be a 1
        The new a1 values at which to evaluate the cubic spline. Must be a 1D array.
    a1_new_batch : jnp.ndarray
        The new a1 values at which to evaluate the cubic spline. Must be a 1D array.
    a2_new_batch : jnp.ndarray
        The new a2 values at which to evaluate the cubic spline. Must be a 1D array.
        
    Returns
    -------
    jnp.ndarray
        A 1D array of interpolated values at the new coordinates (q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch).

    """

    # find indices and conditions for q, mtot, a1, a2 
    n = q_new_batch.shape[0]
    q_array_batch = np.empty((n, 4))
    q_condition_i_batch = np.empty(n, dtype=np.int32)
    m_array_batch = np.empty((n, 4))
    m_condition_i_batch = np.empty(n, dtype=np.int32)
    a1_array_batch = np.empty((n, 4))
    a1_condition_i_batch = np.empty(n, dtype=np.int32)
    a2_array_batch = np.empty((n, 4))
    a2_condition_i_batch = np.empty(n, dtype=np.int32)
    # # becareful, cube_4x4x4x4 can be memory intensive
    cube_4x4x4x4 = np.empty((n, 4, 4, 4, 4)) # reduced snrpartialscaled_array

    for i in prange(n):
        q_idx, q_condition_i_batch[i] = find_index_1d_numba(q_array, q_new_batch[i])
        m_idx, m_condition_i_batch[i] = find_index_1d_numba(mtot_array, mtot_new_batch[i])
        a1_idx, a1_condition_i_batch[i] = find_index_1d_numba(a1_array, a1_new_batch[i])
        a2_idx, a2_condition_i_batch[i] = find_index_1d_numba(a2_array, a2_new_batch[i])

        # Fill the batch arrays with the 4 points around the index
        q_array_batch[i, :] = q_array[q_idx-1:q_idx+3]
        m_array_batch[i, :] = mtot_array[m_idx-1:m_idx+3]
        a1_array_batch[i, :] = a1_array[a1_idx-1:a1_idx+3]
        a2_array_batch[i, :] = a2_array[a2_idx-1:a2_idx+3]

        # Fill the 4D cube with the corresponding values from snrpartialscaled_array
        cube_4x4x4x4[i] = snrpartialscaled_array[q_idx-1:q_idx+3, m_idx-1:m_idx+3, a1_idx-1:a1_idx+3, a2_idx-1:a2_idx+3]


    out = np.zeros(n)  # Output array for the interpolated values
    for i in prange(n):
        out[i] = spline_interp_4x4x4x4pts_numba(
            # axis arrays with 4 elements
            q_array_batch[i], 
            m_array_batch[i], 
            a1_array_batch[i], 
            a2_array_batch[i],
            # 4D array of snrpartialscaled values with shape (4, 4, 4, 4)
            cube_4x4x4x4[i],
            # new coordinates for interpolation
            q_new_batch[i], 
            mtot_new_batch[i], 
            a1_new_batch[i], 
            a2_new_batch[i],
            # conditions for interpolation
            q_condition_i_batch[i], 
            m_condition_i_batch[i], 
            a1_condition_i_batch[i], 
            a2_condition_i_batch[i]
        )
    return out

@njit
def get_interpolated_snr_aligned_spins_numba(
    mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time,
    ra, dec, a_1, a_2, detector_tensor, snr_partialscaled,
    ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000
):
    
    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # These calculations are performed once on the full arrays
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)
    Mc = ((mass_1 * mass_2)**(3. / 5.)) / ((mass_1 + mass_2)**(1. / 5.))
    A1 = Mc**(5.0 / 6.0)
    ci_2 = np.cos(theta_jn)**2
    ci_param = ((1 + np.cos(theta_jn)**2) / 2)**2

    # Pre-allocate full result arrays
    snr_partial_ = np.zeros((len_, size))
    d_eff = np.zeros((len_, size))
    snr = np.zeros((len_, size))

    # Loop over each detector
    for j in range(len_):
        # Iterate over the data in batches
        for i in range(0, size, batch_size):
            # Define the start and end indices for the current batch
            start_idx = i
            end_idx = min(i + batch_size, size)

            # Call the spline function with a slice (batch) of the input data
            snr_partial_[j, start_idx:end_idx] = spline_interp_4x4x4x4pts_batched_numba(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                a1_array=a1_arr,
                a2_array=a_2_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_new_batch=ratio[start_idx:end_idx],
                mtot_new_batch=mtot[start_idx:end_idx],
                a1_new_batch=a_1[start_idx:end_idx],
                a2_new_batch=a_2[start_idx:end_idx],
            )
        
        # These calculations use the fully populated snr_partial_[j, :] array for the current detector
        # and are performed after all batches for this detector are processed.
        d_eff[j, :] = luminosity_distance / np.sqrt(Fp[j]**2 * ci_param + Fc[j]**2 * ci_2)
        snr[j, :] = snr_partial_[j, :] * A1 / d_eff[j, :]

    snr_effective = np.sqrt(np.sum(snr**2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

######### Interpolation 2D #########
@njit
def spline_interp_4x4pts_numba(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new, int_q, int_m):

    partialsnr_along_q = np.zeros(4)
    for i in range(4):
        # Get the 4 y-points for mtot interpolation directly from the main array
        partialsnr_along_m = np.empty(4)
        for j in range(4):
            partialsnr_along_m[j] = snrpartialscaled_array[i, j]

        partialsnr_along_q[i] = spline_interp_4pts_numba(
            x_eval=mtot_new, # point at which partialSNR is calculated
            x_pts=mtot_array,
            y_pts=partialsnr_along_m,
            condition_i=int_m
        )
    final_snr = spline_interp_4pts_numba(
        q_new,
        q_array,
        partialsnr_along_q,
        int_q
    )

    return final_snr

@njit(parallel=True)
def spline_interp_4x4pts_batched_numba(q_array, mtot_array, snrpartialscaled_array, q_new_batch, mtot_new_batch):

    # find indices and conditions for q, mtot, a1, a2 
    n = q_new_batch.shape[0]
    q_array_batch = np.empty((n, 4))
    q_condition_i_batch = np.empty(n, dtype=np.int32)
    m_array_batch = np.empty((n, 4))
    m_condition_i_batch = np.empty(n, dtype=np.int32)
    # # becareful, cube_4x4 can be memory intensive
    cube_4x4x4x4 = np.empty((n, 4, 4)) # reduced snrpartialscaled_array

    for i in prange(n):
        q_idx, q_condition_i_batch[i] = find_index_1d_numba(q_array, q_new_batch[i])
        m_idx, m_condition_i_batch[i] = find_index_1d_numba(mtot_array, mtot_new_batch[i])

        # Fill the batch arrays with the 4 points around the index
        q_array_batch[i, :] = q_array[q_idx-1:q_idx+3]
        m_array_batch[i, :] = mtot_array[m_idx-1:m_idx+3]

        # Fill the 4D cube with the corresponding values from snrpartialscaled_array
        cube_4x4x4x4[i] = snrpartialscaled_array[q_idx-1:q_idx+3, m_idx-1:m_idx+3]


    out = np.zeros(n)  # Output array for the interpolated values
    for i in prange(n):
        out[i] = spline_interp_4x4pts_numba(
            # axis arrays with 4 elements
            q_array_batch[i], 
            m_array_batch[i], 
            # 4D array of snrpartialscaled values with shape (4, 4, 4, 4)
            cube_4x4x4x4[i],
            # new coordinates for interpolation
            q_new_batch[i], 
            mtot_new_batch[i], 
            # conditions for interpolation
            q_condition_i_batch[i], 
            m_condition_i_batch[i], 
        )
    return out

@njit
def get_interpolated_snr_no_spins_numba(
    mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time,
    ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000
):
    size = mass_1.shape[0]
    len_det = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # Must be njit-compatible; see note below
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3. / 5.)) / ((mass_1 + mass_2) ** (1. / 5.))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = np.cos(theta_jn) ** 2
    ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = np.zeros((len_det, size))
    d_eff = np.zeros((len_det, size))
    snr = np.zeros((len_det, size))

    for j in range(len_det):  # Parallelize over detectors!
        snr_partial_[j] = spline_interp_4x4pts_batched_numba(
            q_array=ratio_arr,
            mtot_array=mtot_arr,
            snrpartialscaled_array=snr_partialscaled[j],
            q_new_batch=ratio,
            mtot_new_batch=mtot,
        )
        d_eff[j] = luminosity_distance / np.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        snr[j] = snr_partial_[j] * A1 / d_eff[j]

    snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))
    return snr, snr_effective, snr_partial_, d_eff