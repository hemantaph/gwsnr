import numpy as np
from numba import njit

@njit
def findchirp_chirptime(m1, m2, fmin):
    """
    Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.
    -----------------
    Input parameters
    -----------------
    m1         : component mass of BBH, m1>m2, unit(Mo)
    m2         : component mass of BBH, m1>m2, unit(Mo)
    fmin       : minimum frequency cut-off for the analysis, unit(s)
    -----------------
    Return values
    -----------------
    chirp_time : Time taken from f_min to f_lso (frequency at last stable orbit), unit(s)
    """

    Gamma = 0.5772156649015329
    Pi = np.pi
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
    ans = m[0,0]*n[0,0] + m[0,1]*n[0,1] + m[0,2]*n[0,2] + m[1,0]*n[1,0] + m[1,1]*n[1,1] + m[1,2]*n[1,2] + m[2,0]*n[2,0] + m[2,1]*n[2,1] + m[2,2]*n[2,2]
    return ans

@njit
def gps_to_gmst(gps_time):
    slope = 7.292115855382993e-05
    time0 = 1126259642.413
    time = gps_time - time0
    return slope*time+36137.068361399164

@njit
def ra_dec_to_theta_phi(ra, dec, gmst):
    phi = ra - gmst
    theta = np.pi / 2.0 - dec
    return theta, phi

@njit
def get_polarization_tensor(ra, dec, time, psi, mode='plus'):
    gmst = np.fmod(gps_to_gmst(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode == 'plus':
        return einsum1(m, m) - einsum1(n, n)
    elif mode == 'cross':
        return einsum1(m, n) + einsum1(n, m)

@njit
def antenna_response(ra, dec, time, psi, detector_tensor, mode='plus'):

    polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode=mode)
    return einsum2(detector_tensor, polarization_tensor)

@njit
def antenna_response_array(ra, dec, time, psi, detector_tensor):

    len_det = len(detector_tensor)
    len_param = len(ra) 
    Fp = np.zeros((len_det, len_param))
    Fc = np.zeros((len_det, len_param))

    for j in range(len_det):
        for i in range(len_param):
            # print("ra", ra[j])
            # print("dec", dec[j])
            # print("time", time[j])
            # print("psi", psi[j])
            # print("detector_tensor", detector_tensor[i])
            Fp[j,i] = antenna_response(ra[i], dec[i], time[i], psi[i], detector_tensor[j], mode="plus")
            Fc[j,i] = antenna_response(ra[i], dec[i], time[i], psi[i], detector_tensor[j], mode="cross")

    return Fp, Fc

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

@njit
def get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_halfscaled, ratio_arr, mtot_arr):
    
    size = len(mass_1)
    len_ = len(detector_tensor)
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = np.cos(theta_jn) ** 2
    ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2
    
    size = len(mass_1)
    snr_half_ = np.zeros((len_,size))
    d_eff = np.zeros((len_,size))
    snr = np.zeros((len_,size))
    for j in range(len_):
        for i in range(size):
            snr_half_coeff = snr_halfscaled[j]
            snr_half_[j,i] = cubic_spline_interpolator2d(mtot[i], ratio[i], snr_half_coeff, mtot_arr, ratio_arr)
            d_eff[j,i] =luminosity_distance[i] / np.sqrt(
                    Fp[j,i]**2 * ci_param[i] + Fc[j,i]**2 * ci_2[i]
                )

    snr = snr_half_ * A1 / d_eff
    snr_effective = np.sqrt(np.sum(snr ** 2, axis=0))

    return snr, snr_effective

@njit
def cubic_spline_interpolator2d(xnew, ynew, coefficients, x, y):

    len_y = len(y)
    y_idx = np.searchsorted(y, ynew)
    # print(y_idx)
    if y_idx-1 <= 0:
        y_idx1 = 0
        y_idx2 = 1
        y_idx3 = 2
        y_idx4 = 4
        coeff_low, coeff_high = 0, 4
        # print("a")
    elif y_idx+1 >= len_y:
        y_idx1 = len_y - 4
        y_idx2 = len_y - 3
        y_idx3 = len_y - 2
        y_idx4 = len_y - 1
        coeff_low, coeff_high = 8, 12
        # print("b")
    else:
        y_idx1 = y_idx - 2
        y_idx2 = y_idx - 1
        y_idx3 = y_idx
        y_idx4 = y_idx + 1
        coeff_low, coeff_high = 4, 8
        # print("c")

    # print(y_idx1, y_idx2, y_idx3, y_idx4)
    # print(len(y))
    y1, y2, y3, y4 = y[y_idx1], y[y_idx2], y[y_idx3], y[y_idx4]
    z1 = cubic_spline_interpolator(xnew, coefficients[y_idx1], x)
    z2 = cubic_spline_interpolator(xnew, coefficients[y_idx2], x)
    z3 = cubic_spline_interpolator(xnew, coefficients[y_idx3], x)
    z4 = cubic_spline_interpolator(xnew, coefficients[y_idx4], x)

    coeff = coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4)
    matrixD = coeff[coeff_low:coeff_high]

    matrixB = np.array([ynew**3, ynew**2, ynew, 1])
    result = np.dot(matrixB, matrixD)

    return result

@njit
def cubic_spline_interpolator(xnew, coefficients, x):
    # Handling extrapolation
    i = np.searchsorted(x, xnew) - 1 if xnew > x[0] else 0

    # Calculate the relative position within the interval
    dx = xnew - x[i]

    # Calculate the interpolated value
    # Cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
    a, b, c, d = coefficients[:, i]
    result = d + c*dx + b*dx**2 + a*dx**3
    return result

@njit
def coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4):
        
    matrixA = np.array([
        [y1**3, y1**2, y1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [y2**3, y2**2, y2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, y2**3, y2**2, y2, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, y3**3, y3**2, y3, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, y3**3, y3**2, y3, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, y4**3, y4**2, y4, 1],
        [3*y2**2, 2*y2, 1, 0, -3*y2**2, -2*y2, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3*y3**2, 2*y3, 1, 0, -3*y3**2, -2*y3, -1, 0],
        [6*y2, 2, 0, 0, -6*y2, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6*y3, 2, 0, 0, -6*y3, -2, 0, 0],
        [6*y1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 6*y4, 2, 0, 0],
    ])
    matrixC = np.array([z1, z2, z2, z3, z3, z4, 0, 0, 0, 0, 0, 0])
    return np.dot(np.linalg.inv(matrixA), matrixC)

