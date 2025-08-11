import numpy as np
from numba import njit, prange
from .njit_functions import antenna_response_array


##############################
@njit
def find_index_1d_numba(x_array, x_new):
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
def cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i):
    """
    Catmull-Rom spline interpolation (Numba-friendly, 4 points).
    x_pts: 4 points, assumed sorted, uniform or non-uniform spacing.
    y_pts: values at those points.
    x_eval: point at which to evaluate.
    condition_i: for edge handling.
    """
    # Handle edges with linear interpolation
    if condition_i == 1:
        return y_pts[0] + (y_pts[1] - y_pts[0]) * (x_eval - x_pts[0]) / (x_pts[1] - x_pts[0])
    elif condition_i == 3:
        return y_pts[2] + (y_pts[3] - y_pts[2]) * (x_eval - x_pts[2]) / (x_pts[3] - x_pts[2])

    # Main Catmull-Rom, between x_pts[1] and x_pts[2]
    x0, x1, x2, x3 = x_pts[0], x_pts[1], x_pts[2], x_pts[3]
    y0, y1, y2, y3 = y_pts[0], y_pts[1], y_pts[2], y_pts[3]

    # Avoid division by zero
    denom = x2 - x1
    # if denom == 0.0:
    #     return y1  # Fallback

    t = (x_eval - x1) / denom

    # Catmull-Rom spline formula
    a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
    b = y0 - 2.5*y1 + 2.0*y2 - 0.5*y3
    c = -0.5*y0 + 0.5*y2
    d = y1

    snr = a*t**3 + b*t**2 + c*t + d
    return snr
    
######### Interpolation 4D #########
@njit
def cubic_spline_4d_numba(
    q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, 
    q_new, mtot_new, a1_new, a2_new
):
    # 1. Find all indices and conditions at once
    q_idx, int_q = find_index_1d_numba(q_array, q_new)
    m_idx, int_m = find_index_1d_numba(mtot_array, mtot_new)
    a1_idx, int_a1 = find_index_1d_numba(a1_array, a1_new)
    a2_idx, int_a2 = find_index_1d_numba(a2_array, a2_new)

    # 2. Perform interpolation without creating a large 'F' slice or intermediate arrays
    # This involves more complex indexing but avoids memory allocation.
    
    vals_after_a1 = np.zeros(4)
    for i in range(4): # Loop over q-dimension index offset
        vals_after_m = np.zeros(4)
        for j in range(4): # Loop over mtot-dimension index offset
            vals_after_a2 = np.zeros(4)
            for k in range(4): # Loop over a1-dimension index offset
                # Get the 4 y-points for a2 interpolation directly from the main array
                # y_pts_a2 = snrpartialscaled_array[q_idx-1+i, m_idx-1+j, a1_idx-1+k, a2_idx-1:a2_idx+3]
                # Numba doesn't like fancy indexing, so build array manually:
                y_pts_a2 = np.array([snrpartialscaled_array[q_idx-1+i, m_idx-1+j, a1_idx-1+k, a2_idx-1+l] for l in range(4)])
                # Interpolate for a2
                vals_after_a2[k] = cubic_function_4pts_numba(a2_new, a2_array[a2_idx-1:a2_idx+3], y_pts_a2, int_a2)
            
            vals_after_m[j] = cubic_function_4pts_numba(
                a1_new, 
                a1_array[a1_idx-1:a1_idx+3], 
                vals_after_a2, 
                int_a1
            )
        
        vals_after_a1[i] = cubic_function_4pts_numba(
            mtot_new, 
            mtot_array[m_idx-1:m_idx+3], 
            vals_after_m, 
            int_m
        )
        
    final_snr = cubic_function_4pts_numba(
        q_new, 
        q_array[q_idx-1:q_idx+3], 
        vals_after_a1, 
        int_q
    )
    
    return final_snr

@njit(parallel=True)
def cubic_spline_4d_batched_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_batch, mtot_batch, a1_batch, a2_batch):
    n = q_batch.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        out[i] = cubic_spline_4d_numba(
            q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array,
            q_batch[i], mtot_batch[i], a1_batch[i], a2_batch[i]
        )
    return out

@njit
def get_interpolated_snr_aligned_spins_numba(
    mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time,
    ra, dec, a_1, a_2, detector_tensor, snr_partialscaled,
    ratio_arr, mtot_arr, a1_arr, a_2_arr
):
    
    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # Numba doesn't support non-numba functions; you must jit this separately
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3. / 5.)) / ((mass_1 + mass_2) ** (1. / 5.))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = np.cos(theta_jn) ** 2
    ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = np.zeros((len_, size))
    d_eff = np.zeros((len_, size))
    snr = np.zeros((len_, size))

    for j in range(len_):
        snr_partial_[j, :] = cubic_spline_4d_batched_numba(
            q_array=ratio_arr,
            mtot_array=mtot_arr,
            a1_array=a1_arr,
            a2_array=a_2_arr,
            snrpartialscaled_array=snr_partialscaled[j],
            q_batch=ratio,
            mtot_batch=mtot,
            a1_batch=a_1,
            a2_batch=a_2,
        )
        # Effective distance per sample
        d_eff[j, :] = luminosity_distance / np.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        snr[j, :] = snr_partial_[j, :] * A1 / d_eff[j, :]

    snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

######### Interpolation 2D #########
@njit
def cubic_spline_2d_numba(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new):
    q_idx, int_q = find_index_1d_numba(q_array, q_new)
    m_idx, int_m = find_index_1d_numba(mtot_array, mtot_new)

    partialsnr_q = np.zeros(4)
    for i in range(4):
        y_pts_mtot = np.array([snrpartialscaled_array[q_idx - 1 + i, m_idx - 1 + j] for j in range(4)])
        partialsnr_q[i] = cubic_function_4pts_numba(mtot_new, mtot_array[m_idx - 1:m_idx + 3], y_pts_mtot, int_m)
    final_snr = cubic_function_4pts_numba(q_new, q_array[q_idx - 1:q_idx + 3], partialsnr_q, int_q)

    return final_snr

@njit(parallel=True)
def cubic_spline_2d_batched_numba(q_array, mtot_array, snrpartialscaled_array, q_batch, mtot_batch):
    n = q_batch.shape[0]
    out = np.zeros(n)
    for i in prange(n):  # Parallel over samples
        out[i] = cubic_spline_2d_numba(
            q_array, mtot_array, snrpartialscaled_array,
            q_batch[i], mtot_batch[i]
        )
    return out

@njit
def get_interpolated_snr_no_spins_numba(
    mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time,
    ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr
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
        snr_partial_[j, :] = cubic_spline_2d_batched_numba(
            ratio_arr,
            mtot_arr,
            snr_partialscaled[j],
            ratio,
            mtot,
        )
        d_eff[j, :] = luminosity_distance / np.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        snr[j, :] = snr_partial_[j, :] * A1 / d_eff[j, :]

    snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))
    return snr, snr_effective, snr_partial_, d_eff

# @njit
# def cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i):
#     """
#     Numba-compatible cubic interpolation using 4 points.
#     """

#     if condition_i == 0:
#         # linear interpolation using first and second points
#         snr = y_pts[0] + (y_pts[1] - y_pts[0]) * (x_eval - x_pts[0]) / (x_pts[1] - x_pts[0])
#         return snr
#     elif condition_i == 1:
#         # linear interpolation using 3rd and 4th points
#         snr = y_pts[2] + (y_pts[3] - y_pts[2]) * (x_eval - x_pts[2]) / (x_pts[3] - x_pts[2])
#         return snr
#     else:
#         # Build the linear system as in your JAX code
#         x = x_pts
#         y = y_pts

#         matrixA = np.array([
#             [x[0]**3, x[0]**2, x[0], 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [x[1]**3, x[1]**2, x[1], 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, x[1]**3, x[1]**2, x[1], 1.0, 0, 0, 0, 0],
#             [0, 0, 0, 0, x[2]**3, x[2]**2, x[2], 1.0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, x[2]**3, x[2]**2, x[2], 1.0],
#             [0, 0, 0, 0, 0, 0, 0, 0, x[3]**3, x[3]**2, x[3], 1.0],
#             [3*x[1]**2, 2*x[1], 1.0, 0, -3*x[1]**2, -2*x[1], -1.0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 3*x[2]**2, 2*x[2], 1.0, 0, -3*x[2]**2, -2*x[2], -1.0, 0],
#             [6*x[1], 2.0, 0.0, 0.0, -6*x[1], -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 6*x[2], 2.0, 0.0, 0.0, -6*x[2], -2.0, 0.0, 0.0],
#             [6*x[0], 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 6*x[3], 2.0, 0, 0],
#         ])
#         matrixC = np.array([
#             y[0], y[1], y[1], y[2], y[2], y[3],
#             0, 0, 0, 0, 0, 0
#         ])
#         # Rest are already zero

#         coeffs = np.linalg.solve(matrixA, matrixC)
#         # Take coefficients for the 2nd cubic (index 4:7)
#         snr = coeffs[4]*x_eval**3 + coeffs[5]*x_eval**2 + coeffs[6]*x_eval + coeffs[7]
#         return snr

# @njit
# def get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr):
#     """
#     Function to calculate the interpolated snr for a given set of parameters

#     Parameters
#     ----------
#     mass_1 : `numpy.ndarray`
#         Mass of the first body in solar masses.
#     mass_2 : `numpy.ndarray`
#         Mass of the second body in solar masses.
#     luminosity_distance : `float`
#         Luminosity distance to the source in Mpc.
#     theta_jn : `numpy.ndarray`
#         Angle between the total angular momentum and the line of sight to the source in radians.
#     psi : `numpy.ndarray`
#         Polarization angle of the source.
#     geocent_time : `numpy.ndarray`
#         GPS time of the source.
#     ra : ``numpy.ndarray`
#         Right ascension of the source in radians.
#     dec : `numpy.ndarray`
#         Declination of the source in radians.
#     detector_tensor : array-like
#         Detector tensor for the detector (3x3 matrix)
#     snr_partialscaled : `numpy.ndarray`
#         Array of snr_partialscaled coefficients for the detector.
#     ratio_arr : `numpy.ndarray`
#         Array of mass ratio values for the snr_partialscaled coefficients.
#     mtot_arr : `numpy.ndarray`
#         Array of total mass values for the snr_partialscaled coefficients.
    
#     Returns
#     -------
#     snr : `float`
#         snr of the detector.
#     """

#     size = len(mass_1)
#     len_ = len(detector_tensor)
#     mtot = mass_1 + mass_2
#     ratio = mass_2 / mass_1
#     # get array of antenna response
#     Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

#     Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
#     A1 = Mc ** (5.0 / 6.0)
#     ci_2 = np.cos(theta_jn) ** 2
#     ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2
    
#     size = len(mass_1)
#     snr_partial_ = np.zeros((len_,size))
#     d_eff = np.zeros((len_,size))
#     snr = np.zeros((len_,size))
#     # loop over the detectors
#     for j in range(len_):
#         # loop over the parameter points
#         for i in range(size):
#             snr_partial_[j,i] = cubic_spline_interpolator2d(mtot[i], ratio[i], snr_partialscaled[j], mtot_arr, ratio_arr)
#             d_eff[j,i] =luminosity_distance[i] / np.sqrt(
#                     Fp[j,i]**2 * ci_param[i] + Fc[j,i]**2 * ci_2[i]
#                 )

#     snr = snr_partial_ * A1 / d_eff
#     snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))

#     return snr, snr_effective, snr_partial_, d_eff

# @njit
# def cubic_spline_interpolator2d(xnew, ynew, coefficients, x, y):
#     """
#     Function to calculate the interpolated value of snr_partialscaled given the mass ratio (ynew) and total mass (xnew). This is based off 2D bicubic spline interpolation.

#     Parameters
#     ----------
#     xnew : `float`
#         Total mass of the binary.
#     ynew : `float`
#         Mass ratio of the binary.
#     coefficients : `numpy.ndarray`
#         Array of coefficients for the cubic spline interpolation.
#     x : `numpy.ndarray`
#         Array of total mass values for the coefficients.
#     y : `numpy.ndarray`
#         Array of mass ratio values for the coefficients.

#     Returns
#     -------
#     result : `float`
#         Interpolated value of snr_partialscaled.
#     """

#     len_y = len(y)
#     # find the index nearest to the ynew in y
#     y_idx = np.searchsorted(y, ynew) - 1 if ynew > y[0] else 0

#     if (ynew>y[0]) and (ynew<y[1]): # if ynew is between the first two points
#         if ynew > y[y_idx] + (y[y_idx+1] - y[y_idx]) / 2: # if ynew is closer to the second point
#             y_idx = y_idx + 1 # move to the second point
#         result = cubic_spline_interpolator(xnew, coefficients[y_idx], x)
#     elif y_idx == 0:  # lower end point
#         result = cubic_spline_interpolator(xnew, coefficients[0], x)
#         # print("a")
#     elif y_idx+1 == len_y:  # upper end point
#         result = cubic_spline_interpolator(xnew, coefficients[-1], x)
#         # print("b")
#     elif y_idx+2 == len_y:  # upper end point
#         result = cubic_spline_interpolator(xnew, coefficients[-1], x)
#         # print("b")
#     else:
#         y_idx1 = y_idx - 1
#         y_idx2 = y_idx
#         y_idx3 = y_idx + 1
#         y_idx4 = y_idx + 2
#         coeff_low, coeff_high = 4, 8
#         # print("c")
#         y1, y2, y3, y4 = y[y_idx1], y[y_idx2], y[y_idx3], y[y_idx4]
#         z1 = cubic_spline_interpolator(xnew, coefficients[y_idx1], x)
#         z2 = cubic_spline_interpolator(xnew, coefficients[y_idx2], x)
#         z3 = cubic_spline_interpolator(xnew, coefficients[y_idx3], x)
#         z4 = cubic_spline_interpolator(xnew, coefficients[y_idx4], x)

#         coeff = coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4)
#         matrixD = coeff[coeff_low:coeff_high]
#         matrixB = np.array([ynew**3, ynew**2, ynew, 1])
#         result = np.dot(matrixB, matrixD)

#     return result

# @njit
# def cubic_spline_interpolator(xnew, coefficients, x):
#     """
#     Function to calculate the interpolated value of snr_partialscaled given the total mass (xnew). This is based off 1D cubic spline interpolation.

#     Parameters
#     ----------
#     xnew : `float`
#         Total mass of the binary.
#     coefficients : `numpy.ndarray`
#         Array of coefficients for the cubic spline interpolation.
#     x : `numpy.ndarray`
#         Array of total mass values for the coefficients.

#     Returns
#     -------
#     result : `float`
#         Interpolated value of snr_partialscaled.
#     """
#     # Handling extrapolation
#     i = np.searchsorted(x, xnew) - 1 if xnew > x[0] else 0

#     # Calculate the relative position within the interval
#     dx = xnew - x[i]

#     # Calculate the interpolated value
#     # Cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
#     a, b, c, d = coefficients[:, i]
#     result = d + c*dx + b*dx**2 + a*dx**3
#     return result

# @njit
# def coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4):
#     """
#     Function to generate the coefficients for the cubic spline interpolation of fn(y)=z.

#     Parameters
#     ----------
#     y1, y2, y3, y4, z1, z2, z3, z4: `float`
#         Values of y and z for the cubic spline interpolation.

#     Returns
#     -------
#     coefficients: `numpy.ndarray`
#         Coefficients for the cubic spline interpolation.
#     """
#     matrixA = np.array([
#         [y1**3, y1**2, y1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [y2**3, y2**2, y2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, y2**3, y2**2, y2, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, y3**3, y3**2, y3, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, y3**3, y3**2, y3, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, y4**3, y4**2, y4, 1],
#         [3*y2**2, 2*y2, 1, 0, -3*y2**2, -2*y2, -1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 3*y3**2, 2*y3, 1, 0, -3*y3**2, -2*y3, -1, 0],
#         [6*y2, 2, 0, 0, -6*y2, -2, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 6*y3, 2, 0, 0, -6*y3, -2, 0, 0],
#         [6*y1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 6*y4, 2, 0, 0],
#     ])
#     matrixC = np.array([z1, z2, z2, z3, z3, z4, 0, 0, 0, 0, 0, 0])
#     return np.dot(np.linalg.inv(matrixA), matrixC)

# @njit
# def linear_interpolator(xnew, coefficients, x, bounds_error=False, fill_value=None):
#     """
#     """

#     idx_max = len(x)-1
#     if bounds_error:
#         if (xnew < x[0]) or (xnew > x[idx_max]):
#             raise ValueError("Chosen x values out of bound")

#     # Handling extrapolation
#     i = np.searchsorted(x, xnew) - 1 
#     idx1 = xnew <= x[0]
#     i[idx1] = 0
#     idx2 = xnew > x[idx_max]
#     i[idx2] = idx_max - 1

#     # Calculate the relative position within the interval
#     dx = xnew - x[i]

#     # Calculate the interpolated value
#     # linear polynomial: a + b*dx 
#     const, slope = coefficients[i].T
#     ynew = const + slope*dx

#     if fill_value is not None:
#         ynew[idx1] = fill_value
#         ynew[idx2] = fill_value

#     return ynew

# @njit
# def coefficients_generator_linear(x, y):
#     """
#     """

#     lenx = len(x)
#     x2 = x[1:lenx]
#     x1 = x[0:lenx-1]
#     y2 = y[1:lenx]
#     y1 = y[0:lenx-1]

#     slope = (y2-y1)/(x2-x1)
#     const = y1

#     return const,slope