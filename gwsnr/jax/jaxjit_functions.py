import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, vmap

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
def einsum1(m, n):
    # Outer product: 3x1, 3x1 -> 3x3
    return jnp.outer(m, n)

@jit
def einsum2(m, n):
    # Elementwise product and sum: trace(m*n.T) if both are 3x3
    return jnp.sum(m * n)

@jit
def gps_to_gmst(gps_time):
    slope = 7.292115855382993e-05
    time0 = 1126259642.413
    time = gps_time - time0
    return slope * time + 36137.068361399164

@jit
def ra_dec_to_theta_phi(ra, dec, gmst):
    phi = ra - gmst
    theta = jnp.pi / 2.0 - dec
    return theta, phi

@jit
def get_polarization_tensor_plus(ra, dec, time, psi):
    gmst = jnp.fmod(gps_to_gmst(time), 2 * jnp.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = jnp.array([jnp.cos(phi) * jnp.cos(theta), jnp.cos(theta) * jnp.sin(phi), -jnp.sin(theta)])
    v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)

    return einsum1(m, m) - einsum1(n, n)
    
def get_polarization_tensor_cross(ra, dec, time, psi, mode='plus'):
    gmst = jnp.fmod(gps_to_gmst(time), 2 * jnp.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = jnp.array([jnp.cos(phi) * jnp.cos(theta), jnp.cos(theta) * jnp.sin(phi), -jnp.sin(theta)])
    v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)

    return einsum1(m, n) + einsum1(n, m)

@jit
def antenna_response_plus(ra, dec, time, psi, detector_tensor):
    polarization_tensor = get_polarization_tensor_plus(ra, dec, time, psi)
    return einsum2(detector_tensor, polarization_tensor)

@jit
def antenna_response_cross(ra, dec, time, psi, detector_tensor):
    polarization_tensor = get_polarization_tensor_cross(ra, dec, time, psi)
    return einsum2(detector_tensor, polarization_tensor)

@jit
def antenna_response_array(ra, dec, time, psi, detector_tensor):
    # detector_tensor shape: (n_det, 3, 3)
    # ra, dec, time, psi shape: (n_param,)
    n_det = detector_tensor.shape[0]
    n_param = ra.shape[0]

    # VMAP over detector and parameter axes
    # Outputs shape (n_det, n_param)
    Fp = vmap(
        lambda d: vmap(
            lambda ra_i, dec_i, time_i, psi_i: antenna_response_plus(
                ra_i, dec_i, time_i, psi_i, d
            )
        )(ra, dec, time, psi)
    )(detector_tensor)
    Fc = vmap(
        lambda d: vmap(
            lambda ra_i, dec_i, time_i, psi_i: antenna_response_cross(
                ra_i, dec_i, time_i, psi_i, d
            )
        )(ra, dec, time, psi)
    )(detector_tensor)
    return Fp, Fc